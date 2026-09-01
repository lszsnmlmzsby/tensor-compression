from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.benchmark_tensor_qwen_inference import (  # noqa: E402
    aggregate_cost_reports,
    apply_checkpoint_architecture,
    apply_checkpoint_dataset_policy,
    attention_score_proxy,
    distribution_summary,
    paired_contingency,
    parse_args,
    parse_methods,
    percentile,
    state_cluster_bootstrap,
)
from scripts.train_tensor_qwen_cross_attention import (  # noqa: E402
    CHECKPOINT_TYPE,
    CHECKPOINT_VERSION,
)


def prediction(
    qa_id: str,
    *,
    state: str,
    task: str,
    correct: bool,
) -> dict:
    return {
        "qa_id": qa_id,
        "state_ref": state,
        "task_type": task,
        "field": "Vx",
        "answer": "A",
        "prediction": "A" if correct else "B",
        "correct": correct,
    }


class TestTensorQwenInferenceBenchmark(unittest.TestCase):
    def test_percentiles_use_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0], 0.95), 1.0)
        self.assertEqual(percentile([0.0, 10.0], 0.25), 2.5)
        summary = distribution_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertEqual(summary["max"], 4.0)

    def test_attention_proxy_counts_padded_execution_shape(self) -> None:
        result = attention_score_proxy(
            batch_size=4,
            padded_tokens=200,
            qwen_layers=48,
            qwen_heads=40,
            dense_bridges=3,
            bridge_heads=8,
            memory_cells=256,
        )
        self.assertEqual(
            result["self_attention_score_elements"],
            4 * 48 * 40 * 200 * 200,
        )
        self.assertEqual(
            result["dense_cross_attention_score_elements"],
            4 * 3 * 8 * 200 * 256,
        )
        self.assertEqual(
            result["total_attention_score_elements"],
            result["self_attention_score_elements"]
            + result["dense_cross_attention_score_elements"],
        )

    def test_cost_aggregation_uses_slowest_rank_and_all_devices(self) -> None:
        base = {
            "records": 2,
            "batches": 1,
            "prompt_lengths": [3, 4],
            "useful_prompt_tokens": 7,
            "padded_prompt_tokens": 8,
            "h2d_payload_bytes": 128,
            "serialized_matrix_utf8_bytes": 32,
            "matrix_values_or_memory_cells": 512,
            "self_attention_score_elements": 100,
            "dense_cross_attention_score_elements": 20,
            "total_attention_score_elements": 120,
            "accelerator_batch_ms": [10.0],
            "cpu_phase_seconds": {},
            "memory": {"peak_allocated_bytes": 1000},
        }
        reports = [
            {**base, "rank": 0, "repetition_wall_seconds": [1.5]},
            {**base, "rank": 1, "repetition_wall_seconds": [2.0]},
        ]
        result = aggregate_cost_reports(reports, expected_records=4)
        self.assertEqual(result["critical_path_wall_seconds"], 2.0)
        self.assertEqual(result["records_per_second"], 2.0)
        self.assertEqual(result["gpu_seconds"], 4.0)
        self.assertEqual(result["prompt_tokens"]["total_useful"], 14)
        self.assertEqual(result["prompt_tokens"]["total_padded"], 16)
        self.assertEqual(result["logical_attention_score_proxy"]["total_attention_score_elements"], 240)

    def test_method_order_protects_unmodified_serialized_qwen(self) -> None:
        self.assertEqual(parse_methods("serialized,dense"), ["serialized", "dense"])
        self.assertEqual(parse_methods(["dense"]), ["dense"])
        with self.assertRaisesRegex(ValueError, "serialized input before"):
            parse_methods("dense,serialized")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_methods("dense,dense")

    def test_cli_paths_override_migrated_config_without_rewriting_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "benchmark.yaml"
            config.write_text(
                "\n".join(
                    [
                        "storage:",
                        f"  output_root: {root / 'outputs'}",
                        "model:",
                        "  name_or_path: org/original-model",
                        "data:",
                        f"  qa_dir: {root / 'old_qa'}",
                        f"  latent_dir: {root / 'old_latents'}",
                        f"  alignment_checkpoint: {root / 'old_alignment.pt'}",
                        "benchmark:",
                        "  methods: [serialized]",
                    ]
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(config),
                    "--model-name-or-path",
                    "org/override-model",
                    "--qa-dir",
                    str(root / "new_qa"),
                    "--latent-dir",
                    str(root / "new_latents"),
                    "--alignment-checkpoint",
                    str(root / "new_alignment.pt"),
                ]
            )
            self.assertEqual(args.model_name_or_path, "org/override-model")
            self.assertEqual(Path(args.qa_dir), root / "new_qa")
            self.assertEqual(Path(args.latent_dir), root / "new_latents")
            self.assertEqual(
                Path(args.qa_alignment_checkpoint), root / "new_alignment.pt"
            )

    def test_checkpoint_architecture_is_the_structure_source_of_truth(self) -> None:
        args = argparse.Namespace()
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE,
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": {
                "layers_1based": [8, 20, 32],
                "bridge_dim": 512,
                "heads": 8,
                "dropout": 0.0,
                "gate_init": 0.0,
                "value_fourier_bands": 4,
                "value_hidden_dim": 128,
                "freeze_spatial_backbone": True,
            },
        }
        observed = apply_checkpoint_architecture(args, checkpoint)
        self.assertEqual(args.cross_attention_layers, [8, 20, 32])
        self.assertEqual(args.bridge_dim, 512)
        self.assertEqual(args.bridge_heads, 8)
        self.assertEqual(args.value_fourier_bands, 4)
        self.assertTrue(args.freeze_spatial_backbone)
        self.assertEqual(args.latent_channel_policy, "all")
        self.assertEqual(observed["latent_channel_policy"], "all")
        self.assertEqual(observed["value_hidden_dim"], 128)

        value_only_checkpoint = {
            **checkpoint,
            "architecture": {
                **checkpoint["architecture"],
                "latent_channel_policy": "value_only",
            },
        }
        value_only_args = argparse.Namespace()
        observed_value_only = apply_checkpoint_architecture(
            value_only_args, value_only_checkpoint
        )
        self.assertEqual(value_only_args.latent_channel_policy, "value_only")
        self.assertEqual(observed_value_only["latent_channel_policy"], "value_only")

    def test_dense_dataset_policy_is_applied_before_first_latent_access(self) -> None:
        dataset = argparse.Namespace(_latent_cache={}, latent_channel_policy="all")
        apply_checkpoint_dataset_policy(dataset, "value_only")
        self.assertEqual(dataset.latent_channel_policy, "value_only")

        accessed_dataset = argparse.Namespace(
            _latent_cache={"state": object()}, latent_channel_policy="all"
        )
        with self.assertRaisesRegex(RuntimeError, "accessed before"):
            apply_checkpoint_dataset_policy(accessed_dataset, "value_only")

    def test_paired_contingency_and_cluster_bootstrap(self) -> None:
        serialized = [
            prediction("s1_a", state="s1", task="task_a", correct=True),
            prediction("s1_b", state="s1", task="task_b", correct=False),
            prediction("s2_a", state="s2", task="task_a", correct=False),
            prediction("s2_b", state="s2", task="task_b", correct=False),
        ]
        dense = [
            prediction("s1_a", state="s1", task="task_a", correct=True),
            prediction("s1_b", state="s1", task="task_b", correct=True),
            prediction("s2_a", state="s2", task="task_a", correct=False),
            prediction("s2_b", state="s2", task="task_b", correct=True),
        ]
        contingency = paired_contingency(serialized, dense)
        self.assertEqual(contingency["both_correct"], 1)
        self.assertEqual(contingency["dense_only_correct"], 2)
        self.assertEqual(contingency["serialized_only_correct"], 0)
        self.assertEqual(contingency["both_wrong"], 1)
        self.assertEqual(contingency["dense_minus_serialized_accuracy"], 0.5)
        with self.assertRaisesRegex(ValueError, "duplicate qa_id"):
            paired_contingency(serialized + [serialized[0]], dense)

        first = state_cluster_bootstrap(serialized, dense, samples=100, seed=7)
        second = state_cluster_bootstrap(serialized, dense, samples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["cluster_count"], 2)
        self.assertEqual(first["micro_accuracy_delta"]["point_estimate"], 0.5)
        self.assertEqual(first["macro_task_accuracy_delta"]["point_estimate"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
