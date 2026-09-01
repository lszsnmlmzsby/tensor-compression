from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.compare_tensor_qwen_stage1_ablation import CONDITIONS, compare_runs, json_digest


def _artifact(condition: str, macro: float, task_a: float, task_b: float) -> dict:
    policy = "value_only" if condition == "no_learned_stage1" else "all"
    condition_index = CONDITIONS.index(condition)
    evaluation_contract = {
        "noninferiority_margin_percentage_points": 1.5,
        "maximum_per_task_drop_percentage_points": 3.0,
        "dense_screening_max_updates": 3000,
    }
    source_tree = {
        "git_available": True,
        "commit": "abc123",
        "tracked_dirty": False,
        "tracked_diff_sha256": "clean",
    }
    stage1_contract = {
        "contract_version": 3,
        "condition": condition,
        "seed": 42,
        "source_stage1_sha256": "c" * 64,
        "source_adapter_state_sha256": "a" * 64,
        "source_encoder_trained_during_alignment": True,
        "source_encoder_origin": "checkpoint",
        "source_encoder_checkpoint": "/immutable/compressor.pt",
        "source_encoder_lineage_complete": True,
        "source_encoder_state_sha256": "0" * 64,
        "initial_adapter_state_sha256": (
            "a" * 64 if condition == "full_stage1_reference" else "b" * 64
        ),
        "value_channel_index": 0,
        "value_channel_contract": {"input_channels": 1},
        "evaluation_contract_sha256": "d" * 64,
        "evaluation_contract": evaluation_contract,
        "latent_channel_policy": policy,
        "dense_forwarded_cli": [],
        "test_protocol_lock": None,
        "direct_qa_checkpoint_sha256": "e" * 64,
        "direct_run_audit": {
            "artifacts": {
                "run_summary_sha256": "8" * 64,
                "run_timing_sha256": "9" * 64,
                "data_audit_sha256": "f" * 64,
                "qa_metadata_audit_sha256": "1" * 64,
                "direct_checkpoint_sha256": "e" * 64,
            },
            "invariants": {
                # The selected epoch is deliberately different: it is an
                # outcome, not a controlled factor.
                "checkpoint_epoch": condition_index + 1,
                "summary": {
                    "train_records": 300,
                    "val_records": 100,
                    "test_records": 100,
                    "total_optimizer_updates": 1200,
                },
                "checkpoint_args": {"seed": 42, "shuffle_seed": 42},
                "adapter_state_schema": {
                    "projection.weight": {"shape": [8, 8], "dtype": "torch.float32"}
                },
                "data_audit": {
                    split: {"records": records, "record_contract_sha256": "2" * 64}
                    for split, records in (("train", 300), ("val", 100), ("test", 100))
                },
                "latent_contract": {"format": "tensor_patch_latent_v1"},
            },
        },
        "traceability": {
            "base_config_sha256": "direct-config",
            "source_tree": source_tree,
        },
        "dense_traceability": {
            "base_config_sha256": "dense-config",
            "source_tree": source_tree,
        },
    }
    by_task = {
        "task_a": {"total": 50, "accuracy": task_a},
        "task_b": {"total": 50, "accuracy": task_b},
    }
    summary = {
        "status": "complete",
        "global_step": 3000,
        "planned_updates": 3000,
        "train_records": 300,
        "val_records": 100,
        "test_records": 100,
        "run_contract": {
            "architecture": {
                "format": "dense",
                "latent_shape": [8, 16, 16],
                "latent_channel_policy": policy,
                "initializer": {
                    "path": f"/{condition}.pt",
                    "sha256": condition,
                    "latent_channel_policy": policy,
                    "source_model": "Qwen/Qwen2.5-14B-Instruct",
                },
                "stage1_ablation": stage1_contract,
            },
            "qa_metadata": {"format": "matched"},
            "install": {"layers": [8, 20, 32], "latent_channel_policy": policy},
            "parameters": {"trainable_parameters": 123},
            "distributed": {"planned_updates": 3000, "effective_batch_size": 12},
            "optimizer": {"lr": 1e-4},
        },
        "final_val": {
            "records": 100,
            "modes": {
                "correct": {
                    "accuracy": macro,
                    "macro_accuracy": macro,
                    "by_task": by_task,
                }
            },
        },
        "final_test": None,
    }
    return {
        "summary": summary,
        "summary_sha256": "4" * 64,
        "data_audit_sha256": "5" * 64,
        "best_checkpoint_sha256": "6" * 64,
        "last_checkpoint_sha256": "7" * 64,
        "data_audit": {
            "splits": {
                split: {
                    "records": records,
                    "record_contract_sha256": "3" * 64,
                }
                for split, records in (("train", 300), ("val", 100), ("test", 100))
            }
        },
    }


def _locked_test_artifacts(tmp_path: Path) -> tuple[dict, Path]:
    validation_artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    validation_result = compare_runs(validation_artifacts)
    comparison_path = tmp_path / "stage1_validation_comparison.json"
    comparison_path.write_text(
        json.dumps(validation_result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lock = {
        "sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "validation_comparison": str(comparison_path),
        "validation_comparison_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "validation_comparison_payload_sha256": json_digest(validation_result),
        "source_stage1_sha256": "c" * 64,
        "evaluation_contract_sha256": "d" * 64,
    }
    test_artifacts = copy.deepcopy(validation_artifacts)
    for condition, artifact in test_artifacts.items():
        summary = artifact["summary"]
        summary["final_test"] = copy.deepcopy(summary["final_val"])
        summary["resume"] = {"sha256": artifact["last_checkpoint_sha256"]}
        contract = summary["run_contract"]["architecture"]["stage1_ablation"]
        contract["dense_forwarded_cli"] = ["--evaluate-test"]
        contract["test_protocol_lock"] = copy.deepcopy(lock)
        # These files are rewritten during the resumed test-only invocation.
        artifact["summary_sha256"] = "0" * 64
        artifact["last_checkpoint_sha256"] = "1" * 64
    return test_artifacts, comparison_path


def test_matched_comparison_applies_preregistered_noninferiority_guards() -> None:
    artifacts = {
        "full_stage1_reference": _artifact("full_stage1_reference", 0.90, 0.90, 0.90),
        "adapter_only": _artifact("adapter_only", 0.89, 0.90, 0.88),
        "no_learned_stage1": _artifact("no_learned_stage1", 0.86, 0.90, 0.82),
    }

    result = compare_runs(artifacts)

    assert result["valid_matched_comparison"] is True
    assert result["comparisons_to_reference"]["adapter_only"][
        "passes_preregistered_screen"
    ] is True
    assert result["comparisons_to_reference"]["no_learned_stage1"][
        "passes_preregistered_screen"
    ] is False
    assert {
        "direct_run_summary_sha256",
        "direct_run_timing_sha256",
        "direct_data_audit_sha256",
        "direct_qa_metadata_audit_sha256",
        "direct_qa_checkpoint_sha256",
    } <= set(result["artifacts"]["full_stage1_reference"])


def test_exact_margin_boundaries_pass_despite_binary_float_roundoff() -> None:
    artifacts = {
        "full_stage1_reference": _artifact("full_stage1_reference", 0.90, 0.90, 0.90),
        # 1.5 pp macro drop and exactly 3.0 pp on task_a.
        "adapter_only": _artifact("adapter_only", 0.885, 0.87, 0.90),
        "no_learned_stage1": _artifact("no_learned_stage1", 0.90, 0.90, 0.90),
    }

    comparison = compare_runs(artifacts)["comparisons_to_reference"]["adapter_only"]

    assert comparison["macro_noninferior"] is True
    assert comparison["all_tasks_within_guardrail"] is True
    assert comparison["passes_preregistered_screen"] is True


def test_macro_drop_just_over_margin_fails() -> None:
    artifacts = {
        "full_stage1_reference": _artifact("full_stage1_reference", 0.90, 0.90, 0.90),
        "adapter_only": _artifact("adapter_only", 0.88499999, 0.88499999, 0.88499999),
        "no_learned_stage1": _artifact("no_learned_stage1", 0.90, 0.90, 0.90),
    }

    comparison = compare_runs(artifacts)["comparisons_to_reference"]["adapter_only"]

    assert comparison["macro_noninferior"] is False
    assert comparison["passes_preregistered_screen"] is False


def test_task_drop_just_over_margin_fails() -> None:
    artifacts = {
        "full_stage1_reference": _artifact("full_stage1_reference", 0.90, 0.90, 0.90),
        # The task average remains exactly 0.885, but task_a is just beyond 3 pp.
        "adapter_only": _artifact("adapter_only", 0.885, 0.86999999, 0.90000001),
        "no_learned_stage1": _artifact("no_learned_stage1", 0.90, 0.90, 0.90),
    }

    comparison = compare_runs(artifacts)["comparisons_to_reference"]["adapter_only"]

    assert comparison["macro_noninferior"] is True
    assert comparison["all_tasks_within_guardrail"] is False
    assert comparison["passes_preregistered_screen"] is False


def test_comparison_rejects_different_qa_record_fingerprint() -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    changed = copy.deepcopy(artifacts)
    changed["adapter_only"]["data_audit"]["splits"]["val"][
        "record_contract_sha256"
    ] = "different"

    with pytest.raises(ValueError, match="QA record audit"):
        compare_runs(changed)


def test_comparison_rejects_different_direct_run_invariants() -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    artifacts["adapter_only"]["summary"]["run_contract"]["architecture"][
        "stage1_ablation"
    ]["direct_run_audit"]["invariants"]["summary"]["total_optimizer_updates"] = 1199

    with pytest.raises(ValueError, match="Direct run invariants"):
        compare_runs(artifacts)


def test_comparison_rejects_direct_audit_checkpoint_sha_mismatch() -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    artifacts["adapter_only"]["summary"]["run_contract"]["architecture"][
        "stage1_ablation"
    ]["direct_run_audit"]["artifacts"]["direct_checkpoint_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checkpoint SHA differs"):
        compare_runs(artifacts)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01, 1.01])
def test_comparison_rejects_invalid_metric_values(invalid: float) -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    artifacts["adapter_only"]["summary"]["final_val"]["modes"]["correct"][
        "macro_accuracy"
    ] = invalid

    with pytest.raises(ValueError, match=r"finite and lie in \[0, 1\]"):
        compare_runs(artifacts)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.01])
def test_comparison_rejects_invalid_margins(invalid: float) -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    for artifact in artifacts.values():
        contract = artifact["summary"]["run_contract"]["architecture"]["stage1_ablation"]
        contract["evaluation_contract"][
            "noninferiority_margin_percentage_points"
        ] = invalid

    with pytest.raises(ValueError, match="margin must be finite and non-negative"):
        compare_runs(artifacts)


def test_comparison_rejects_task_totals_that_do_not_cover_records() -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    artifacts["adapter_only"]["summary"]["final_val"]["modes"]["correct"][
        "by_task"
    ]["task_a"]["total"] = 49

    with pytest.raises(ValueError, match="task totals do not sum to records"):
        compare_runs(artifacts)


def test_comparison_validates_overall_correct_count_contract() -> None:
    artifacts = {
        condition: _artifact(condition, 0.90, 0.90, 0.90) for condition in CONDITIONS
    }
    for artifact in artifacts.values():
        correct = artifact["summary"]["final_val"]["modes"]["correct"]
        correct["correct"] = 90
        correct["total"] = 100
    artifacts["adapter_only"]["summary"]["final_val"]["modes"]["correct"][
        "correct"
    ] = 89

    with pytest.raises(ValueError, match="accuracy is inconsistent with correct/total"):
        compare_runs(artifacts)


def test_test_comparison_accepts_the_locked_dense_and_direct_lineage(
    tmp_path: Path,
) -> None:
    artifacts, _comparison_path = _locked_test_artifacts(tmp_path)

    result = compare_runs(artifacts, split="test")

    assert result["valid_matched_comparison"] is True
    assert result["metric_split"] == "test"


@pytest.mark.parametrize("target", ["dense_data_audit", "direct_data_audit"])
def test_test_comparison_rejects_lineage_changed_after_validation_lock(
    tmp_path: Path,
    target: str,
) -> None:
    artifacts, _comparison_path = _locked_test_artifacts(tmp_path)
    for artifact in artifacts.values():
        if target == "dense_data_audit":
            artifact["data_audit_sha256"] = "2" * 64
        else:
            contract = artifact["summary"]["run_contract"]["architecture"][
                "stage1_ablation"
            ]
            contract["direct_run_audit"]["artifacts"]["data_audit_sha256"] = "2" * 64

    with pytest.raises(ValueError, match="does not use the locked validation"):
        compare_runs(artifacts, split="test")


def test_test_comparison_rejects_modified_locked_comparison_file(tmp_path: Path) -> None:
    artifacts, comparison_path = _locked_test_artifacts(tmp_path)
    comparison_path.write_text(
        comparison_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed after test unlock"):
        compare_runs(artifacts, split="test")
