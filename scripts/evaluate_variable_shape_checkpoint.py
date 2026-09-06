"""Evaluate a variable-grid checkpoint on separately specified, immutable QA.

This is inference, not resume: reconstruct the saved architecture and validate
its initializer, then bind a separate evaluation contract. Never write weights,
optimizer state, best-checkpoint selection, or the original training run summary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import train_tensor_qwen_cross_attention as trainer
from tensor_compression.downstream.variable_shape import experiment_config, mixed_protocol, parse_shapes, record_shape
from tensor_compression.utils.pipeline_config import load_yaml_mapping, resolve_path_string


def training_identity_audit(checkpoint, training_qa_dir, dataset):
    """Bind IDs to the actual checkpoint-hashed train.jsonl, not editable metadata."""
    source = Path(training_qa_dir) / f"{checkpoint['args']['train_split']}.jsonl"
    expected = checkpoint["architecture"]["input_contract"]["qa_output_split_sha256"][checkpoint["args"]["train_split"]]
    digest, samples, seeds = hashlib.sha256(), set(), set()
    with source.open("rb") as reader:
        for line in reader:
            digest.update(line)
            row = json.loads(line)
            samples.add(int(row["sample_index"]))
            synthetic = row.get("metadata", {}).get("synthetic")
            if synthetic:
                seeds.add(int(synthetic["seed"]))
    if digest.hexdigest() != expected:
        raise ValueError("Training QA differs from the checkpoint digest; cannot certify evaluation isolation.")
    eval_samples = {int(row["sample_index"]) for row in dataset.records}
    eval_seeds = {int(row["metadata"]["synthetic"]["seed"]) for row in dataset.records if row["metadata"].get("synthetic")}
    if samples & eval_samples or seeds & eval_seeds:
        raise ValueError("Independent evaluation overlaps checkpoint training trajectories or synthetic seeds.")
    return {"training_jsonl_sha256": expected, "training_samples": len(samples),
            "evaluation_samples": len(eval_samples), "overlapping_samples": 0, "overlapping_synthetic_seeds": 0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="Evaluation dataset configuration; model settings come from checkpoint.")
    parser.add_argument("--profile", choices=["pilot", "full"], required=True)
    parser.add_argument("--qa-dir", required=True)
    parser.add_argument("--training-qa-dir", help="Original training QA directory if relocated.")
    parser.add_argument("--output-dir", required=True, help="A new empty inference output directory.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--model-dir")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.eval_batch_size <= 0 or cli.num_workers < 0:
        raise ValueError("Evaluation batch size must be positive and workers nonnegative.")
    checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**copy.deepcopy(checkpoint["args"]))
    if args.shape_mode != "variable" or args.input_source != "raw_hdf5" or args.memory_init_mode != "scratch":
        raise ValueError("Independent inference supports raw variable-grid scratch checkpoints only.")
    training_qa_dir = cli.training_qa_dir or args.qa_dir
    args.model_name_or_path = cli.model_dir or os.environ.get("FIELD_TO_LLM_MODEL_DIR") or args.model_name_or_path
    args.hf_home = os.environ.get("FIELD_TO_LLM_HF_HOME") or args.hf_home
    args.cache_dir = args.hf_home
    args.device, args.num_workers, args.eval_batch_size = cli.device, cli.num_workers, cli.eval_batch_size
    args.llm_gradient_checkpointing = False
    args.raw_config = experiment_config(load_yaml_mapping(cli.config), cli.profile)
    args.qa_dir = str(Path(cli.qa_dir).expanduser().resolve())
    args.experiment_profile = cli.profile
    args.train_shapes = parse_shapes(args.raw_config["data"]["train_shapes"], allow_odd=mixed_protocol(args.raw_config))
    args.hdf5_path = resolve_path_string(cli.hdf5_path or os.environ.get("PDEBENCH_HDF5") or args.raw_config["data"]["hdf5_path"], ROOT)
    trainer.apply_runtime_environment(args)
    device = trainer.initialize_distributed_device(args.device, distributed_timeout_seconds=args.distributed_timeout_seconds)
    dataset = None
    try:
        output = Path(cli.output_dir).expanduser().resolve()
        def create_output():
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Use a new empty inference output directory: {output}.")
            output.mkdir(parents=True, exist_ok=True)
        trainer.run_on_rank_zero_and_broadcast(create_output, "inference output directory")
        metadata, contract = trainer.run_on_rank_zero_and_broadcast(lambda: trainer.load_metadata_and_contract(args), "independent QA contract")
        shapes = contract["train_shapes"] + contract["heldout_shapes"] + contract["extrapolation_shapes"]
        dataset = trainer.RawHDF5TensorReadoutQADataset(
            Path(args.qa_dir) / f"{cli.split}.jsonl", hdf5_path=args.hdf5_path, patch_size=args.patch_size,
            normalization=contract["normalization"], normalized_dtype=args.raw_normalized_dtype,
            max_records=None, subset_mode="hash_state", subset_seed=args.shuffle_seed,
            shuffle_seed=args.shuffle_seed, input_cache_size=args.latent_cache_size, dynamic_grid=True, allowed_shapes=shapes)
        allowed = set(metadata["trajectory_splits"][cli.split])
        if any(int(row["sample_index"]) not in allowed for row in dataset.records):
            raise ValueError("Evaluation records differ from their declared trajectory partition.")
        if {record_shape(row) for row in dataset.records} != set(map(tuple, shapes)):
            raise ValueError("Evaluation shape coverage differs from metadata.")
        isolation = trainer.run_on_rank_zero_and_broadcast(lambda: training_identity_audit(checkpoint, training_qa_dir, dataset), "training/evaluation isolation")
        data_audit = trainer.run_on_rank_zero_and_broadcast(lambda: trainer.audit_general_qa_datasets({cli.split: dataset}, True), "inference QA audit")
        raw_audit = trainer.run_on_rank_zero_and_broadcast(lambda: trainer.audit_raw_hdf5_input_content({cli.split: dataset}), "inference raw field audit")
        tokenizer = trainer.load_tokenizer(args)
        prompt = trainer.run_on_rank_zero_and_broadcast(lambda: trainer.audit_prompt_tokenization(
            {cli.split: dataset}, tokenizer, max_prompt_tokens=args.max_prompt_tokens,
            prompt_template=args.prompt_template, audit_local_conditioning_prompt=False), "inference prompt audit")
        choices = trainer.run_on_rank_zero_and_broadcast(lambda: trainer.audit_choice_tokenization({cli.split: dataset}, tokenizer), "inference choice audit")
        if not prompt["all_prompts_fit"] or not choices["all_labels_single_token"]:
            raise ValueError("Independent evaluation prompts/choices fail the frozen model's input contract.")
        llm, dtype = trainer.load_llm_with_bounded_host_memory(args, device)
        trainer.seed_everything(args.seed)
        encoder, spatial, memory_shape, initializer = trainer.build_scratch_memory_components(args, (1, args.patch_size, args.patch_size))
        sidecar, _ = trainer.build_sidecar(llm, spatial, args, device, encoder)
        initializer["field_spatial_sha256"] = initializer["sha256"]
        initializer["sha256"] = trainer._module_state_sha256({"full_sidecar": sidecar})
        expected = trainer.architecture_contract(args, (1, args.patch_size, args.patch_size), memory_shape,
            llm.get_input_embeddings().embedding_dim, checkpoint["architecture"]["input_contract"], initializer)
        expected["training_recipe"] = checkpoint["architecture"].get("training_recipe")
        trainer.validate_checkpoint_contract(checkpoint, expected)
        trainer.load_trainable_state_dict(sidecar, checkpoint["trainable_state_dict"])
        metrics = trainer.evaluate(llm, sidecar, tokenizer, dataset, device, dtype, args, ["correct"],
                                   prediction_prefix=output / "predictions" / cli.split)
        def save():
            trainer.atomic_dump_json(output / "metrics.json", metrics)
            trainer.atomic_dump_json(output / "evaluation_contract.json", {
                "format": "independent_variable_grid_evaluation_v1", "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": trainer.sha256_file(checkpoint_path), "checkpoint_step": checkpoint["progress"]["global_step"],
                "split": cli.split, "evaluation_input_contract": contract, "isolation": isolation,
                "data_audit": data_audit, "raw_audit": raw_audit, "prompt_audit": prompt,
                "choice_audit": choices, "training_architecture": checkpoint["architecture"]})
        trainer.run_on_rank_zero_and_broadcast(save, "independent evaluation results")
        trainer.print_eval_summary("independent_eval", metrics)
    finally:
        if dataset is not None:
            dataset.close()
        if trainer.distributed_is_initialized():
            trainer.dist.destroy_process_group()


if __name__ == "__main__":
    main()
