from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from scripts.build_variable_shape_qa import build_dataset
from scripts.build_tensor_patch_matched_qa import evaluation_record_replay
from scripts.mixed_shape_qa import synthetic_raw
from scripts.analyze_field_predictions import analyze
from scripts import evaluate_variable_shape_checkpoint as inference
from scripts import train_tensor_qwen_cross_attention as trainer
from tensor_compression.downstream.variable_shape import experiment_config, parse_shapes, record_shape, ShapeBatchSampler
from tensor_compression.downstream.field_diagnostics import coordinate_bucket, prediction_record, write_prediction_shard
from tensor_compression.utils.pipeline_config import load_yaml_mapping
from test_variable_shape_fields import TinyTokenizer, tiny_qwen

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def mixed_assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed_fields")
    hdf5 = root / "real.hdf5"
    rng = np.random.default_rng(101)
    with h5py.File(hdf5, "w") as handle:
        for field in ("Vx", "Vy", "density", "pressure"):
            handle.create_dataset(field, data=rng.normal(size=(40, 2, 24, 24)).astype("float32"))
    config = experiment_config(load_yaml_mapping(ROOT / "configs/field_to_llm_variable_shape_mixed.yaml"), "pilot")
    config["data"].update(train_shapes=["8x9", "9x8"], heldout_shapes=["9x11"], extrapolation_shapes=["12x12"])
    config["generation"].update(train_states=64, eval_states_per_shape=16, region_size=2)
    config["evaluation"]["screening_records"] = 80
    output = root / "qa"
    metadata = build_dataset(config, output, hdf5)
    return root, hdf5, output, metadata, config


def mixed_args(assets):
    root, hdf5, output, _, config = assets
    with patch.dict("os.environ", {"FIELD_TO_LLM_ROOT": str(root), "PDEBENCH_HDF5": str(hdf5)}), patch.object(
        sys, "argv", ["train", "--config", str(ROOT / "configs/field_to_llm_variable_shape_mixed.yaml"),
                      "--profile", "pilot", "--qa-dir", str(output), "--hdf5-path", str(hdf5)]
    ):
        args = trainer.parse_args()
    args.raw_config = copy.deepcopy(config)
    args.train_shapes = [(8, 9), (9, 8)]
    args.screening_records = 80
    args.num_workers = 0
    args.spatial_adapter_config = {"adapter_dim": 16, "adapter_layers": 1, "adapter_heads": 4}
    args.cross_attention_layers = [1, 2]
    args.bridge_dim, args.bridge_heads = 16, 4
    args.value_hidden_dim = 16
    args.gate_init = 0.1
    args.llm_gradient_checkpointing = False
    args.console_progress = False
    args.torch_dtype = "float32"
    return args


def test_larger_budget_and_geometry_contract():
    config = experiment_config(load_yaml_mapping(ROOT / "configs/field_to_llm_variable_shape_mixed.yaml"), "full")
    shapes = parse_shapes(config["data"]["train_shapes"], allow_odd=True)
    assert len(shapes) == 32 and all(8 <= min(s) and max(s) <= 96 and s[0] * s[1] <= 2048 for s in shapes)
    assert (17, 90) not in shapes and (90, 17) not in shapes
    assert any(h % 2 and w % 2 for h, w in shapes)
    assert config["generation"]["train_states"] == 65536
    assert config["training"]["max_updates"] == 131072
    assert config["generation"]["train_states"] * 9 == 589824
    assert config["evaluation"]["batch_size"] == 1
    with pytest.raises(ValueError):
        parse_shapes(["17x90"])


def test_mixed_replay_mixtures_and_split_integrity(mixed_assets):
    args = mixed_args(mixed_assets)
    metadata, contract = trainer.load_metadata_and_contract(args)
    datasets = trainer.build_datasets(args, contract)
    assert metadata["format"] == "mixed_shape_field_qa_v2"
    assert contract["synthetic_asset_sha256"] == metadata["synthetic_asset"]["sha256"]
    audits = trainer.audit_general_qa_datasets(dict(zip(("train", "val", "test"), datasets[:3])), True)
    assert not any(item["sample_overlap_count"] for item in audits["overlaps"].values())
    for dataset in datasets:
        counts = Counter()
        for record in dataset.records:
            z = dataset.load_latent_for_record(record)[0]
            assert evaluation_record_replay(record, z, numeric_gap=0.1, region_gap=0.2)["eligible"]
            counts[record["metadata"]["source_kind"]] += 1
            if record["task_type"] == "extreme_quadrant":
                assert f"Top means rows 1 through {z.shape[0] // 2}" in record["query"]
            diagnostic = record["metadata"]["diagnostic"]
            if "target_z" in diagnostic:
                target = diagnostic["target"]
                assert diagnostic["target_z"] == float(z[target["row"], target["col"]])
            if record["metadata"]["source_kind"] != "real":
                assert record["field"] == "scalar" and record["sample_index"] >= 10**9
                spec = record["metadata"]["synthetic"]
                raw = synthetic_raw(record_shape(record), record["metadata"]["source_kind"], spec["seed"])
                assert torch.equal(raw, dataset._read_raw_patch(record)[0])
        assert counts["real"] == 2 * counts["correlated"] == 2 * counts["iid"]
        dataset.close()
    train = datasets[0]
    numeric = [r for r in train.records if r["task_type"] == "normalized_point_value"]
    assert Counter(r["metadata"]["numeric_recipe"] for r in numeric) == {"matched": 96, "uniform": 96}
    spatial_counts = Counter((r["metadata"]["source_kind"], r["field"], r["metadata"]["numeric_recipe"], r["task_type"])
                             for r in train.records if r["task_type"] in {"point_compare", "region_mean_compare"})
    for (source, field, recipe, task), count in spatial_counts.items():
        assert count == spatial_counts[source, field, recipe, "region_mean_compare" if task == "point_compare" else "point_compare"]
    for row in numeric:
        assert bool(row["matched_group"]["margin_group_id"]) == (row["metadata"]["numeric_recipe"] == "matched")
    # Synthetic seeds/state namespaces are disjoint, while real trajectory split seed stays v1's.
    samples = list(range(40))
    random.Random(20260905).shuffle(samples)
    assert metadata["real_trajectory_splits"]["train"] == sorted(samples[:32])
    assert len(datasets[3]) == 80
    samplers = [ShapeBatchSampler(train, 3, rank=r, num_replicas=3, training=True) for r in range(3)]
    assert all(s.padding_records_per_epoch == 0 for s in samplers)
    assert Counter(i for s in samplers for batch in s for i in batch) == Counter(range(len(train)))


def test_reproducible_builder_and_asset_tamper(mixed_assets):
    root, hdf5, output, metadata, config = mixed_assets
    repeated = build_dataset(config, root / "repeat", hdf5)
    assert repeated["output_split_sha256"] == metadata["output_split_sha256"]
    args = mixed_args(mixed_assets)
    args.qa_dir = str(root / "repeat")
    with h5py.File(root / "repeat" / "synthetic_fields.hdf5", "r+") as handle:
        handle["train/8x9"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="immutable QA hash"):
        trainer.load_metadata_and_contract(args)


def test_odd_thin_and_extrapolation_memory(mixed_assets):
    args = mixed_args(mixed_assets)
    encoder, spatial, _, _ = trainer.build_scratch_memory_components(args, (1, 17, 90))
    memory = trainer.DenseTensorMemory(spatial, 2, 16, False, encoder)
    for height, width in ((17, 90), (90, 17), (21, 95), (95, 21), (64, 64), (16, 128), (128, 16)):
        x = torch.randn(1, 1, height, width)
        with torch.no_grad():
            assert torch.equal(encoder(x)[:, :1], x)
            encoded = memory(x)
            assert encoded.content.shape == (1, height * width, 16)
            assert torch.isfinite(encoded.content).all()
    x = torch.randn(3, 1, 17, 90)
    encoded = memory(x)
    (encoded.content.square().mean() + encoded.value.square().mean() + encoded.reconstruction_loss).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())


def test_numeric_pairs_across_unequal_rank_shards(mixed_assets):
    records = [json.loads(line) for line in (mixed_assets[2] / "val.jsonl").read_text().splitlines()]
    prefix = mixed_assets[0] / "sharded" / "val"
    shards = []
    for rank in range(3):
        rows = []
        for index in range(rank, len(records), 3):
            record = records[index]
            prediction = record["answer"]
            # Exactly one raw answer is wrong; the paired norm question is on another rank.
            if index == 1:
                prediction = next(label for label in record["choices"] if label != prediction)
            rows.append(prediction_record(record, prediction))
        shards.append(write_prediction_shard(prefix, "correct", rows, rank))
    assert len({shard["records"] for shard in shards}) == 2
    path = prefix.parent / "val.correct.manifest.json"
    path.write_text(json.dumps({"records": len(records), "shards": shards}), encoding="utf-8")
    result = analyze(path)
    assert result["numeric_pairs"]["total"] == 64
    assert result["numeric_pairs"]["norm_correct_raw_wrong"] == 1
    assert result["numeric_pairs"]["both_correct"] == 63
    assert result["numeric_pairs"]["same_value_rank"] == 63


def test_coordinate_diagnostics_cover_region_extent():
    assert coordinate_bucket({"type": "point", "row": 31, "col": 31}, 32) == "le_32"
    assert coordinate_bucket({"type": "point", "row": 0, "col": 32}, 32) == "gt_32"
    assert coordinate_bucket({"type": "region_pair", "a": [0, 0, 4, 4], "b": [30, 0, 4, 4]}, 32) == "gt_32"
    assert coordinate_bucket({"type": "none"}, 96) == "global"


def test_training_resume_predictions_and_diagnostics(mixed_assets):
    args = mixed_args(mixed_assets)
    args.device = "cpu"
    args.output_root = str(mixed_assets[0] / "runs")
    args.run_name = "mixed_smoke"
    args.max_updates, args.epochs = 2, 1
    args.screening_fractions = [0.5, 1.0]
    args.save_every_updates = 1
    args.serialize_llm_loading = False
    def run():
        with patch.object(trainer, "parse_args", return_value=args), patch.object(trainer, "load_tokenizer", return_value=TinyTokenizer()), patch.object(
            trainer, "load_llm_with_bounded_host_memory", side_effect=lambda *a, **kw: (tiny_qwen(), torch.float32)):
            trainer.main()
    with patch.object(trainer.StopController, "distributed_reason", side_effect=[None, "signal"]):
        run()
    run_dir = next(Path(args.output_root).iterdir())
    args.resume = str(run_dir / "cross_attention_last.pt")
    run()
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["global_step"] == 2 and summary["training_budget_completed"]
    result = analyze(run_dir / "predictions" / "final_val.correct.manifest.json")
    assert result["records"] == 320
    assert result["numeric_pairs"]["total"] == 64
    assert result["numeric_pairs"].get("incomplete", 0) == 0
    assert result["breakdowns"]["source:real"]["total"] == 160
    metrics = json.loads((run_dir / "final_val_metrics.json").read_text())["modes"]["correct"]
    assert metrics["correct"] == result["breakdowns"]["all"]["correct"]
    assert sum(item["total"] for item in metrics["by_source"].values()) == 320
    # Standalone inference reads the selected checkpoint without touching training files.
    config_path = mixed_assets[0] / "eval_config.json"
    config_path.write_text(json.dumps({**mixed_assets[4], "profiles": {"pilot": {}}}), encoding="utf-8")
    best_path = run_dir / "cross_attention_best.pt"
    best_hash = trainer.sha256_file(best_path)
    cli = type("CLI", (), dict(checkpoint=str(best_path), config=str(config_path), profile="pilot",
        qa_dir=str(mixed_assets[2]), training_qa_dir=None, output_dir=str(run_dir / "independent"),
        split="val", hdf5_path=str(mixed_assets[1]), model_dir=None, eval_batch_size=2, num_workers=0, device="cpu"))()
    with patch.object(inference, "parse_args", return_value=cli), patch.object(trainer, "load_tokenizer", return_value=TinyTokenizer()), patch.object(
        trainer, "load_llm_with_bounded_host_memory", side_effect=lambda *a, **kw: (tiny_qwen(), torch.float32)):
        inference.main()
    assert trainer.sha256_file(best_path) == best_hash
    assert json.loads((run_dir / "run_summary.json").read_text()) == summary
    independent = analyze(run_dir / "independent" / "predictions" / "val.correct.manifest.json")
    assert independent["records"] == 320
    checkpoint = torch.load(best_path, weights_only=True)
    bad_dataset = type("Dataset", (), {"records": [json.loads((mixed_assets[2] / "train.jsonl").read_text().splitlines()[0])]})()
    with pytest.raises(ValueError, match="overlaps"):
        inference.training_identity_audit(checkpoint, mixed_assets[2], bad_dataset)
    # The same state can be split across DDP ranks: merge by manifest, never rank-local pairs.
    shard = run_dir / "predictions" / "final_val.correct.rank00000.jsonl"
    with shard.open("a", encoding="utf-8") as writer:
        writer.write(shard.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate prediction"):
        analyze(run_dir / "predictions" / "final_val.correct.manifest.json")
