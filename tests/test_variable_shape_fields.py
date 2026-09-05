from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_variable_shape_qa import build_dataset
from scripts.build_tensor_patch_matched_qa import evaluation_record_replay
from scripts.train_tensor_qwen_cross_attention import (
    DenseTensorMemory, build_scratch_memory_components, build_datasets,
    load_metadata_and_contract, parse_args, validate_checkpoint_contract,
    CHECKPOINT_TYPE, CHECKPOINT_VERSION,
    build_sidecar, forward_training_batch, evaluate, main as training_main,
)
from tensor_compression.downstream.patch_qa_prompt import build_prompt
from tensor_compression.downstream.variable_shape import (
    ShapeBatchSampler, experiment_config, record_shape, shape_matched_shuffle_indices,
    rectangular_quadrant,
)
from tensor_compression.utils.pipeline_config import load_yaml_mapping


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("variable_fields")
    hdf5 = root / "fields.hdf5"
    rng = np.random.default_rng(100)
    with h5py.File(hdf5, "w") as handle:
        for field in ("Vx", "density"):
            handle.create_dataset(field, data=rng.normal(size=(40, 3, 24, 24)).astype("float32"))
    config = experiment_config(load_yaml_mapping(ROOT / "configs/field_to_llm_variable_shape.yaml"), "pilot")
    config["data"].update(fields=["Vx", "density"], train_shapes=["4x8", "8x4"],
                          heldout_shapes=["6x8"], extrapolation_shapes=["12x12"])
    config["generation"].update(train_states=32, eval_states_per_shape=8, region_size=2)
    config["evaluation"]["screening_records"] = 60
    output = root / "qa"
    metadata = build_dataset(config, output, hdf5)
    return root, hdf5, output, metadata, config


def configured_args(assets):
    root, hdf5, output, _metadata, config = assets
    with patch.dict("os.environ", {"FIELD_TO_LLM_ROOT": str(root), "PDEBENCH_HDF5": str(hdf5)}), patch.object(
        sys, "argv", ["train", "--config", str(ROOT / "configs/field_to_llm_variable_shape.yaml"),
                      "--profile", "pilot", "--qa-dir", str(output), "--hdf5-path", str(hdf5)]
    ):
        args = parse_args()
    args.train_shapes = [(4, 8), (8, 4)]
    args.raw_config = copy.deepcopy(config)
    args.screening_records = 60
    args.num_workers = 0
    return args


def test_generation_replay_and_split_integrity(assets):
    args = configured_args(assets)
    metadata, contract = load_metadata_and_contract(args)
    datasets = build_datasets(args, contract)
    train, val, test, screen = datasets
    assert len(train) == 32 * 9
    assert len(val) == len(test) == 4 * 8 * 5
    assert len(screen) == 60
    assert {record_shape(row) for row in screen.records} == set(args.train_shapes)
    sample_sets = [{row["sample_index"] for row in data.records} for data in datasets[:3]]
    assert not any(sample_sets[i] & sample_sets[j] for i in range(3) for j in range(i))
    for dataset in datasets:
        for row in dataset.records:
            z = dataset.load_latent_for_record(row)
            assert tuple(z.shape) == (1, *record_shape(row))
            assert evaluation_record_replay(row, z[0], numeric_gap=0.5, region_gap=0.2)["eligible"]
            prompt = build_prompt(row, "field_memory")
            assert "soft tokens" not in prompt
            assert "query_spec" not in prompt
        dataset.close()
    assert metadata["stage1_checkpoint_required"] is False
    assert contract["input_shape"] == [1, -1, -1]
    assert not list(assets[2].glob("*.pt"))
    with pytest.raises(FileExistsError):
        build_dataset(assets[4], assets[2], assets[1])


def test_builder_is_deterministic(assets):
    root, hdf5, _output, metadata, config = assets
    repeated = build_dataset(config, root / "qa_repeat", hdf5)
    assert repeated["output_split_sha256"] == metadata["output_split_sha256"]


def test_shape_sampler_preserves_groups_and_exact_distributed_evaluation(assets):
    args = configured_args(assets)
    _, contract = load_metadata_and_contract(args)
    train, val, test, screen = build_datasets(args, contract)
    samplers = [ShapeBatchSampler(train, 3, rank=rank, num_replicas=3, seed=7, training=True) for rank in range(3)]
    batches = [list(sampler) for sampler in samplers]
    assert len({len(value) for value in batches}) == 1
    for step in zip(*batches):
        assert len({record_shape(train.records[index]) for batch in step for index in batch}) == 1
        for batch in step:
            assert len({train.records[index]["matched_group"]["batch_group_id"] for index in batch}) == 1
            assert [train.records[index]["matched_group"]["batch_member_index"] for index in batch] == [0, 1, 2]
    assert Counter(index for rank in batches for batch in rank for index in batch) == Counter(range(len(train)))
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != batches[0]
    for world in (1, 3, 7):
        covered = []
        for rank in range(world):
            for batch in ShapeBatchSampler(val, 3, rank=rank, num_replicas=world):
                assert len({record_shape(val.records[index]) for index in batch}) == 1
                covered.extend(batch)
        assert Counter(covered) == Counter(range(len(val)))
    for dataset in (train, val, test, screen):
        dataset.close()


def test_shuffle_matches_shape_and_changes_trajectory(assets):
    rows = [json.loads(line) for line in (assets[2] / "val.jsonl").read_text().splitlines()]
    for index, other in enumerate(shape_matched_shuffle_indices(rows, 17)):
        assert record_shape(rows[index]) == record_shape(rows[other])
        assert rows[index]["sample_index"] != rows[other]["sample_index"]


def test_dynamic_memory_preserves_values_and_parameterization():
    config = load_yaml_mapping(ROOT / "configs/field_to_llm_variable_shape.yaml")
    args = SimpleNamespace(patch_size=16, seed=42, shape_mode="fixed", field_encoder_config=config["field_encoder"],
                           spatial_adapter_config={"adapter_dim": 16, "adapter_layers": 1, "adapter_heads": 4})
    torch.manual_seed(3)
    fixed_encoder, fixed_spatial, _, _ = build_scratch_memory_components(args, (1, 16, 16))
    args.shape_mode = "variable"
    torch.manual_seed(3)
    encoder, spatial, _, _ = build_scratch_memory_components(args, (1, 8, 32))
    square = torch.randn(2, 1, 16, 16)
    assert torch.equal(encoder(square), fixed_encoder(square))
    for left, right in zip(spatial.spatial_input_states(encoder(square)), fixed_spatial.spatial_input_states(fixed_encoder(square))):
        assert torch.equal(left, right)
    memory = DenseTensorMemory(spatial, 2, 16, False, encoder)
    count = sum(parameter.numel() for parameter in memory.parameters())
    for height, width in ((8, 16), (16, 8), (32, 32), (40, 40), (48, 48)):
        x = torch.randn(1, 1, height, width)
        encoded = encoder(x)
        assert torch.equal(encoded[:, :1], x)
        output = memory(x)
        assert output.content.shape == (1, height * width, 16)
        assert output.value.shape == output.content.shape
        loss = output.content.square().mean() + output.value.square().mean() + output.reconstruction_loss
        loss.backward()
        assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in memory.parameters())
        assert sum(parameter.numel() for parameter in memory.parameters()) == count
        memory.zero_grad(set_to_none=True)


def test_raw_value_tampering_and_profile_mismatch_rejected(assets):
    args = configured_args(assets)
    _, contract = load_metadata_and_contract(args)
    datasets = build_datasets(args, contract)
    dataset = datasets[0]
    row = copy.deepcopy(dataset.records[0])
    row["metadata"]["normalized_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="QA hash"):
        dataset.load_latent_for_record(row)
    args.experiment_profile = "full"
    with pytest.raises(ValueError, match="profile"):
        load_metadata_and_contract(args)
    for dataset in datasets:
        dataset.close()


def test_rectangular_quadrants_and_explicit_profile():
    assert [rectangular_quadrant(row, col, 8, 32) for row, col in ((0, 0), (0, 16), (4, 0), (7, 31))] == list("ABCD")
    config = load_yaml_mapping(ROOT / "configs/field_to_llm_variable_shape.yaml")
    with pytest.raises(ValueError, match="explicitly"):
        experiment_config(config, None)
    assert experiment_config(config, "pilot")["generation"]["train_states"] * 9 == 12096
    assert experiment_config(config, "full")["generation"]["train_states"] * 9 == 143721


def test_resume_recipe_mismatch_is_rejected():
    architecture = {"qwen_model": "Qwen/Qwen2.5-14B-Instruct", "initializer": {"sha256": "a"},
                    "input_contract": {}, "latent_contract": {}, "shape_mode": "variable",
                    "training_recipe": {"world_size": 3, "planned_updates": 31938, "experiment_profile": "full"}}
    checkpoint = {"checkpoint_type": CHECKPOINT_TYPE, "checkpoint_version": CHECKPOINT_VERSION,
                  "architecture": copy.deepcopy(architecture)}
    architecture["training_recipe"]["planned_updates"] = 1500
    with pytest.raises(ValueError, match="training_recipe"):
        validate_checkpoint_contract(checkpoint, architecture)


class TinyTokenizer:
    """Deterministic test vocabulary; exercises Qwen without downloading weights."""
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [self(item, **kwargs)["input_ids"] for item in text]}
        if text.strip() in "ABCD" and len(text.strip()) == 1:
            return {"input_ids": [2 + "ABCD".index(text.strip())]}
        return {"input_ids": [6 + sum(word.encode()) % 50 for word in text.split()]}

    def encode(self, text, **kwargs):
        return self(text, **kwargs)["input_ids"]


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=1024,
                                      pad_token_id=0, eos_token_id=1, attention_dropout=0.0))


def tiny_args(assets):
    args = configured_args(assets)
    args.spatial_adapter_config = {"adapter_dim": 16, "adapter_layers": 1, "adapter_heads": 4}
    args.cross_attention_layers = [1, 2]
    args.bridge_dim, args.bridge_heads = 16, 4
    args.value_hidden_dim = 16
    args.gate_init = 0.1
    args.llm_gradient_checkpointing = False
    args.console_progress = False
    args.torch_dtype = "float32"
    return args


def test_tiny_qwen_forward_backward_and_shape_metrics(assets):
    args = tiny_args(assets)
    _, contract = load_metadata_and_contract(args)
    train, val, test, screen = build_datasets(args, contract)
    llm = tiny_qwen()
    encoder, spatial, _, _ = build_scratch_memory_components(args, (1, 4, 8))
    sidecar, _ = build_sidecar(llm, spatial, args, torch.device("cpu"), encoder)
    tokenizer = TinyTokenizer()
    for shape in args.train_shapes:
        records = [row for row in train.records if record_shape(row) == shape][:3]
        values = torch.stack([train.load_latent_for_record(row) for row in records])
        loss, _ = forward_training_batch(llm, sidecar, tokenizer, records, values, torch.device("cpu"), torch.float32, args)
        loss.backward()
        sidecar.clear()
        assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in sidecar.parameters() if parameter.requires_grad)
        sidecar.zero_grad(set_to_none=True)
    result = evaluate(llm, sidecar, tokenizer, val, torch.device("cpu"), torch.float32, args, ["correct"])
    metrics = result["modes"]["correct"]
    assert metrics["total"] == len(val)
    assert len(metrics["by_shape"]) == 4
    assert len(metrics["by_shape_task"]) == 20
    assert {key: item["total"] for key, item in metrics["by_shape_partition"].items()} == {"seen": 80, "heldout": 40, "extrapolation": 40}
    assert sum(item["correct"] for item in metrics["by_shape"].values()) == metrics["correct"]
    for dataset in (train, val, test, screen):
        dataset.close()


def test_training_resume_and_evaluate_only_smoke(assets):
    args = tiny_args(assets)
    args.device = "cpu"
    args.output_root = str(assets[0] / "runs")
    args.run_name = "variable_shape_test"
    args.max_updates = 2
    args.epochs = 1
    args.screening_fractions = [0.5, 1.0]
    args.save_every_updates = 1
    args.serialize_llm_loading = False
    import scripts.train_tensor_qwen_cross_attention as trainer
    import scripts.summarize_variable_shape_run as summarizer

    def run():
        with patch.object(trainer, "parse_args", return_value=args), patch.object(
            trainer, "load_tokenizer", return_value=TinyTokenizer()
        ), patch.object(trainer, "load_llm_with_bounded_host_memory", side_effect=lambda *a, **kw: (tiny_qwen(), torch.float32)):
            training_main()

    # Interrupt after one committed update, then continue the same two-update schedule.
    with patch.object(trainer.StopController, "distributed_reason", side_effect=[None, "signal"]):
        run()
    run_dir = next(Path(args.output_root).iterdir())
    interrupted = json.loads((run_dir / "run_summary.json").read_text())
    assert interrupted["global_step"] == 1 and interrupted["status"] == "interrupted_resumable"
    args.resume = str(run_dir / "cross_attention_last.pt")
    run()
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["global_step"] == 2 and summary["training_budget_completed"]
    checkpoint = torch.load(run_dir / "cross_attention_last.pt", weights_only=True)
    args.resume = str(run_dir / "cross_attention_last.pt")
    # A completed run can be resumed and evaluated without changing any weights.
    args.evaluate_only = True
    args.evaluate_test = True
    run()
    resumed = torch.load(run_dir / "cross_attention_last.pt", weights_only=True)
    assert resumed["progress"]["global_step"] == 2
    for key, value in checkpoint["trainable_state_dict"].items():
        assert torch.equal(value, resumed["trainable_state_dict"][key])
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["evaluation_only"]
    assert summary["best_step"] == torch.load(run_dir / "cross_attention_best.pt", weights_only=True)["progress"]["global_step"]
    rows = summarizer.summarize(run_dir, "test", run_dir / "test.csv")
    assert len(rows) == 4
