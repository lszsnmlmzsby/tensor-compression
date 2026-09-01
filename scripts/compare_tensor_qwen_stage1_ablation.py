from __future__ import annotations

"""Audit and compare the three matched Stage-1 necessity conditions."""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CONDITIONS = (
    "full_stage1_reference",
    "adapter_only",
    "no_learned_stage1",
)
RESULT_FORMAT = "stage1_necessity_comparison_v1"
MARGIN_TOLERANCE_PERCENTAGE_POINTS = 1e-9
DIRECT_RUN_ARTIFACT_FIELDS = (
    "run_summary_sha256",
    "run_timing_sha256",
    "data_audit_sha256",
    "qa_metadata_audit_sha256",
    "direct_checkpoint_sha256",
)


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checkpoint_path(run_dir: Path, recorded: Any, fallback_name: str) -> Path:
    raw = Path(str(recorded or fallback_name)).expanduser()
    candidates = [raw] if raw.is_absolute() else [run_dir / raw]
    candidates.append(run_dir / fallback_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        f"Missing dense checkpoint {fallback_name}; tried {[str(value) for value in candidates]}."
    )


def load_run_artifacts(value: str | Path) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    summary_path = path / "run_summary.json" if path.is_dir() else path
    run_dir = summary_path.parent
    data_audit_path = run_dir / "data_audit.json"
    if not summary_path.is_file():
        raise ValueError(f"Missing dense run_summary.json: {summary_path}")
    if not data_audit_path.is_file():
        raise ValueError(f"Missing dense data_audit.json: {data_audit_path}")
    summary = _load_json(summary_path, label="run summary")
    data_audit = _load_json(data_audit_path, label="data audit")
    best_path = _run_checkpoint_path(
        run_dir, summary.get("best_checkpoint"), "cross_attention_best.pt"
    )
    last_path = _run_checkpoint_path(
        run_dir, summary.get("last_checkpoint"), "cross_attention_last.pt"
    )
    return {
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "summary": summary,
        "summary_sha256": _sha256_file(summary_path),
        "data_audit": data_audit,
        "data_audit_sha256": _sha256_file(data_audit_path),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": _sha256_file(last_path),
    }


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid {label} mapping.")
    return dict(value)


def experiment_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    run_contract = _mapping(summary.get("run_contract"), label="run_contract")
    architecture = _mapping(run_contract.get("architecture"), label="architecture")
    return _mapping(architecture.get("stage1_ablation"), label="stage1 comparison contract")


def stable_run_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return every invariant that must match across the three conditions."""

    run_contract = _mapping(summary.get("run_contract"), label="run_contract")
    architecture = _mapping(run_contract.get("architecture"), label="architecture")
    initializer = _mapping(architecture.get("initializer"), label="initializer")
    install = _mapping(run_contract.get("install"), label="install")
    for key in ("path", "sha256", "latent_channel_policy"):
        initializer.pop(key, None)
    install.pop("latent_channel_policy", None)
    for key in ("initializer", "latent_channel_policy", "stage1_ablation"):
        architecture.pop(key, None)
    return {
        "architecture": architecture,
        "initializer_identity": initializer,
        "qa_metadata": run_contract.get("qa_metadata"),
        "install": install,
        "parameters": run_contract.get("parameters"),
        "distributed": run_contract.get("distributed"),
        "optimizer": run_contract.get("optimizer"),
        "planned_updates": int(summary.get("planned_updates", -1)),
        "train_records": int(summary.get("train_records", -1)),
        "val_records": int(summary.get("val_records", -1)),
        "test_records": int(summary.get("test_records", -1)),
    }


def _integer(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not a boolean.")
    try:
        observed = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer: {value!r}.") from error
    if isinstance(value, float) and value != observed:
        raise ValueError(f"{label} must be an exact integer: {value!r}.")
    if observed < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be {qualifier}: {observed}.")
    return observed


def _accuracy(value: Any, *, label: str) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be numeric: {value!r}.") from error
    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
        raise ValueError(f"{label} must be finite and lie in [0, 1]: {observed!r}.")
    return observed


def _metric_payload(summary: Mapping[str, Any], split: str) -> dict[str, Any]:
    key = "final_val" if split == "validation" else "final_test"
    evaluation = _mapping(summary.get(key), label=key)
    modes = _mapping(evaluation.get("modes"), label=f"{key}.modes")
    correct = _mapping(modes.get("correct"), label=f"{key}.modes.correct")
    raw_by_task = _mapping(correct.get("by_task"), label=f"{key}.modes.correct.by_task")
    records = _integer(evaluation.get("records"), label=f"{key}.records", positive=True)
    expected_records_key = "val_records" if split == "validation" else "test_records"
    expected_records = _integer(
        summary.get(expected_records_key), label=expected_records_key, positive=True
    )
    if records != expected_records:
        raise ValueError(
            f"{key}.records differs from summary.{expected_records_key}: "
            f"evaluation={records}, summary={expected_records}."
        )
    accuracy = _accuracy(correct.get("accuracy"), label=f"{key}.modes.correct.accuracy")
    macro_accuracy = _accuracy(
        correct.get("macro_accuracy"), label=f"{key}.modes.correct.macro_accuracy"
    )
    if "correct" in correct or "total" in correct:
        if "correct" not in correct or "total" not in correct:
            raise ValueError(f"{key}.modes.correct must provide both correct and total.")
        correct_total = _integer(
            correct.get("total"), label=f"{key}.modes.correct.total", positive=True
        )
        correct_count = _integer(
            correct.get("correct"), label=f"{key}.modes.correct.correct"
        )
        if correct_total != records:
            raise ValueError(
                f"{key}.modes.correct.total differs from records: "
                f"total={correct_total}, records={records}."
            )
        if correct_count > correct_total:
            raise ValueError(f"{key}.modes.correct.correct exceeds its total.")
        if not math.isclose(
            accuracy,
            correct_count / correct_total,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{key}.modes.correct.accuracy is inconsistent with correct/total."
            )
    if not raw_by_task:
        raise ValueError(f"{key}.modes.correct.by_task must not be empty.")
    by_task: dict[str, dict[str, Any]] = {}
    task_total = 0
    for task, raw_values in raw_by_task.items():
        values = _mapping(raw_values, label=f"{key}.modes.correct.by_task.{task}")
        total = _integer(
            values.get("total"),
            label=f"{key}.modes.correct.by_task.{task}.total",
            positive=True,
        )
        task_accuracy = _accuracy(
            values.get("accuracy"),
            label=f"{key}.modes.correct.by_task.{task}.accuracy",
        )
        normalized = {"total": total, "accuracy": task_accuracy}
        if "correct" in values:
            task_correct = _integer(
                values.get("correct"),
                label=f"{key}.modes.correct.by_task.{task}.correct",
            )
            if task_correct > total:
                raise ValueError(
                    f"{key}.modes.correct.by_task.{task}.correct exceeds its total."
                )
            if not math.isclose(
                task_accuracy,
                task_correct / total,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{key}.modes.correct.by_task.{task}.accuracy is inconsistent "
                    "with correct/total."
                )
            normalized["correct"] = task_correct
        by_task[str(task)] = normalized
        task_total += total
    if task_total != records:
        raise ValueError(
            f"{key} task totals do not sum to records: tasks={task_total}, records={records}."
        )
    task_macro = sum(values["accuracy"] for values in by_task.values()) / len(by_task)
    if not math.isclose(macro_accuracy, task_macro, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{key}.modes.correct.macro_accuracy is inconsistent with by_task: "
            f"reported={macro_accuracy}, recomputed={task_macro}."
        )
    return {
        "records": records,
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "by_task": by_task,
    }


def _require_equal(label: str, values: Mapping[str, Any]) -> None:
    digests = {condition: json_digest(value) for condition, value in values.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"Matched comparison invariant differs for {label}: {digests}.")


def _is_sha256(value: Any) -> bool:
    normalized = str(value or "").lower()
    return len(normalized) == 64 and all(character in "0123456789abcdef" for character in normalized)


def _within_margin(drop: float, margin: float) -> bool:
    return drop <= margin or math.isclose(
        drop,
        margin,
        rel_tol=0.0,
        abs_tol=MARGIN_TOLERANCE_PERCENTAGE_POINTS,
    )


def _direct_run_audit(
    contract: Mapping[str, Any], *, condition: str
) -> tuple[dict[str, Any], dict[str, str]]:
    audit = _mapping(contract.get("direct_run_audit"), label=f"{condition} direct_run_audit")
    invariants = _mapping(
        audit.get("invariants"), label=f"{condition} direct_run_audit.invariants"
    )
    # Best-checkpoint epoch is an outcome of the shared selection rule, not a
    # controlled factor. It can legitimately differ between conditions.
    invariants.pop("checkpoint_epoch", None)
    raw_artifacts = _mapping(
        audit.get("artifacts"), label=f"{condition} direct_run_audit.artifacts"
    )
    artifacts: dict[str, str] = {}
    for field in DIRECT_RUN_ARTIFACT_FIELDS:
        value = str(raw_artifacts.get(field, "")).lower()
        if not _is_sha256(value):
            raise ValueError(f"{condition} Direct run audit has invalid {field}.")
        artifacts[field] = value
    direct_checkpoint_sha = str(contract.get("direct_qa_checkpoint_sha256", "")).lower()
    if artifacts["direct_checkpoint_sha256"] != direct_checkpoint_sha:
        raise ValueError(
            f"{condition} Direct audit checkpoint SHA differs from the dense initializer."
        )
    return invariants, artifacts


def compare_runs(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    split: str = "validation",
) -> dict[str, Any]:
    if tuple(artifacts) != CONDITIONS:
        raise ValueError(f"Expected ordered conditions {CONDITIONS}, got {tuple(artifacts)}.")
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported comparison split: {split!r}.")
    summaries = {
        condition: _mapping(values.get("summary"), label=f"{condition} summary")
        for condition, values in artifacts.items()
    }
    contracts = {condition: experiment_contract(summary) for condition, summary in summaries.items()}
    evaluation_contract = _mapping(
        contracts["full_stage1_reference"].get("evaluation_contract"),
        label="evaluation_contract",
    )
    expected_updates = _integer(
        evaluation_contract.get("dense_screening_max_updates"),
        label="evaluation_contract.dense_screening_max_updates",
        positive=True,
    )
    for condition, summary in summaries.items():
        if summary.get("status") != "complete":
            raise ValueError(f"{condition} run is not complete: status={summary.get('status')!r}.")
        global_step = _integer(
            summary.get("global_step"), label=f"{condition} global_step"
        )
        planned_updates = _integer(
            summary.get("planned_updates"),
            label=f"{condition} planned_updates",
            positive=True,
        )
        if global_step != planned_updates:
            raise ValueError(f"{condition} did not reach its planned optimizer-update budget.")
        if planned_updates != expected_updates:
            raise ValueError(
                f"{condition} planned_updates does not match the preregistered "
                f"budget {expected_updates}."
            )
        if contracts[condition].get("condition") != condition:
            raise ValueError(
                f"Run labeled {condition} contains condition={contracts[condition].get('condition')!r}."
            )
        forwarded = [str(value) for value in contracts[condition].get("dense_forwarded_cli", [])]
        if split == "validation":
            if "final_test" not in summary or summary.get("final_test") is not None:
                raise ValueError(f"{condition} validation run has already accessed the test split.")
            if "test_protocol_lock" not in contracts[condition] or contracts[condition].get(
                "test_protocol_lock"
            ) is not None:
                raise ValueError(f"{condition} validation run has a test protocol lock.")
            if "--evaluate-test" in forwarded:
                raise ValueError(f"{condition} validation run requested test evaluation.")
        elif "--evaluate-test" not in forwarded:
            raise ValueError(f"{condition} test run did not explicitly request test evaluation.")

    invariant_contract_fields = (
        "contract_version",
        "seed",
        "source_stage1_sha256",
        "source_adapter_state_sha256",
        "source_encoder_trained_during_alignment",
        "source_encoder_origin",
        "source_encoder_checkpoint",
        "source_encoder_lineage_complete",
        "source_encoder_state_sha256",
        "value_channel_index",
        "value_channel_contract",
        "evaluation_contract_sha256",
    )
    for field in invariant_contract_fields:
        _require_equal(field, {condition: contract.get(field) for condition, contract in contracts.items()})
    _require_equal(
        "Direct base config",
        {
            condition: _mapping(contract.get("traceability"), label="traceability").get(
                "base_config_sha256"
            )
            for condition, contract in contracts.items()
        },
    )
    _require_equal(
        "dense base config",
        {
            condition: _mapping(
                contract.get("dense_traceability"), label="dense_traceability"
            ).get("base_config_sha256")
            for condition, contract in contracts.items()
        },
    )
    for trace_name in ("traceability", "dense_traceability"):
        _require_equal(
            f"{trace_name} source tree",
            {
                condition: {
                    key: _mapping(
                        _mapping(contract.get(trace_name), label=trace_name).get("source_tree"),
                        label=f"{trace_name}.source_tree",
                    ).get(key)
                    for key in ("git_available", "commit", "tracked_dirty", "tracked_diff_sha256")
                }
                for condition, contract in contracts.items()
            },
        )
        _require_equal(
            f"{trace_name} runtime versions",
            {
                condition: _mapping(contract.get(trace_name), label=trace_name).get(
                    "runtime_versions"
                )
                for condition, contract in contracts.items()
            },
        )
    _require_equal(
        "dense run contract",
        {condition: stable_run_contract(summary) for condition, summary in summaries.items()},
    )
    direct_run_invariants: dict[str, dict[str, Any]] = {}
    direct_run_artifacts: dict[str, dict[str, str]] = {}
    for condition, contract in contracts.items():
        invariants, direct_artifacts = _direct_run_audit(contract, condition=condition)
        direct_run_invariants[condition] = invariants
        direct_run_artifacts[condition] = direct_artifacts
    _require_equal("Direct run invariants", direct_run_invariants)
    _require_equal(
        "QA record audit",
        {condition: values.get("data_audit") for condition, values in artifacts.items()},
    )
    data_audit_splits: dict[str, dict[str, Any]] = {}
    for condition, values in artifacts.items():
        data_audit = _mapping(values.get("data_audit"), label=f"{condition} data_audit")
        splits = _mapping(data_audit.get("splits"), label=f"{condition} data_audit.splits")
        data_audit_splits[condition] = splits
        for split_name in ("train", "val", "test"):
            split_audit = _mapping(
                splits.get(split_name), label=f"{condition} data_audit.splits.{split_name}"
            )
            if not _is_sha256(split_audit.get("record_contract_sha256")):
                raise ValueError(
                    f"{condition} {split_name} data audit lacks a record_contract_sha256."
                )
            audit_records = _integer(
                split_audit.get("records"),
                label=f"{condition} data_audit.splits.{split_name}.records",
                positive=True,
            )
            summary_key = "val_records" if split_name == "val" else f"{split_name}_records"
            summary_records = _integer(
                summaries[condition].get(summary_key),
                label=f"{condition} summary.{summary_key}",
                positive=True,
            )
            if audit_records != summary_records:
                raise ValueError(
                    f"{condition} {split_name} data-audit records differ from the run summary: "
                    f"audit={audit_records}, summary={summary_records}."
                )

    policies = {
        condition: str(contract.get("latent_channel_policy", ""))
        for condition, contract in contracts.items()
    }
    expected_policies = {
        "full_stage1_reference": "all",
        "adapter_only": "all",
        "no_learned_stage1": "value_only",
    }
    if policies != expected_policies:
        raise ValueError(f"Unexpected condition latent-channel policies: {policies}.")
    source_adapter_sha = str(contracts["full_stage1_reference"]["source_adapter_state_sha256"])
    initial_states = {
        condition: str(contract.get("initial_adapter_state_sha256", ""))
        for condition, contract in contracts.items()
    }
    if initial_states["full_stage1_reference"] != source_adapter_sha:
        raise ValueError("Matched reference did not preserve the source Stage-1 adapter state.")
    if (
        not initial_states["adapter_only"]
        or initial_states["adapter_only"] != initial_states["no_learned_stage1"]
        or initial_states["adapter_only"] == source_adapter_sha
    ):
        raise ValueError(
            "Both ablations must share the same deterministic random adapter state, distinct "
            f"from the source: source={source_adapter_sha}, initial={initial_states}."
        )

    locked_validation: dict[str, Any] | None = None
    if split == "test":
        lock_identities = {}
        for condition, contract in contracts.items():
            lock = _mapping(contract.get("test_protocol_lock"), label="test_protocol_lock")
            lock_identities[condition] = {
                key: lock.get(key)
                for key in (
                    "sha256",
                    "payload_sha256",
                    "validation_comparison_sha256",
                    "validation_comparison_payload_sha256",
                    "source_stage1_sha256",
                    "evaluation_contract_sha256",
                )
            }
        _require_equal("test protocol lock", lock_identities)
        reference_lock = _mapping(
            contracts["full_stage1_reference"].get("test_protocol_lock"),
            label="test_protocol_lock",
        )
        for field in (
            "sha256",
            "payload_sha256",
            "validation_comparison_sha256",
            "validation_comparison_payload_sha256",
            "source_stage1_sha256",
            "evaluation_contract_sha256",
        ):
            if not _is_sha256(reference_lock.get(field)):
                raise ValueError(f"Test protocol lock has invalid {field}.")
        comparison_path = Path(str(reference_lock.get("validation_comparison", ""))).expanduser()
        if not comparison_path.is_file():
            raise ValueError(f"Locked validation comparison is missing: {comparison_path}.")
        if _sha256_file(comparison_path) != reference_lock["validation_comparison_sha256"]:
            raise ValueError("Locked validation comparison file changed after test unlock.")
        locked_validation = _load_json(comparison_path, label="locked validation comparison")
        if (
            locked_validation.get("format") != RESULT_FORMAT
            or locked_validation.get("valid_matched_comparison") is not True
            or locked_validation.get("metric_split") != "validation"
            or json_digest(locked_validation)
            != reference_lock["validation_comparison_payload_sha256"]
        ):
            raise ValueError("Test protocol lock does not resolve to the validated comparison payload.")

    metrics = {condition: _metric_payload(summary, split) for condition, summary in summaries.items()}
    audit_split_name = "val" if split == "validation" else "test"
    for condition, metric in metrics.items():
        split_audit = _mapping(
            data_audit_splits[condition].get(audit_split_name),
            label=f"{condition} data_audit.splits.{audit_split_name}",
        )
        audit_records = _integer(
            split_audit.get("records"),
            label=f"{condition} data_audit.splits.{audit_split_name}.records",
            positive=True,
        )
        if metric["records"] != audit_records:
            raise ValueError(
                f"{condition} {split} metrics use {metric['records']} records, "
                f"but the data audit records {audit_records}."
            )
    _require_equal(
        f"{split} record/task denominators",
        {
            condition: {
                "records": metric["records"],
                "task_totals": {
                    task: int(values.get("total", -1))
                    for task, values in metric["by_task"].items()
                },
            }
            for condition, metric in metrics.items()
        },
    )
    artifact_lineage = {
        condition: {
            "run_summary_sha256": values.get("summary_sha256"),
            "data_audit_sha256": values.get("data_audit_sha256"),
            "best_checkpoint_sha256": values.get("best_checkpoint_sha256"),
            "last_checkpoint_sha256": values.get("last_checkpoint_sha256"),
            "direct_qa_checkpoint_sha256": contracts[condition].get(
                "direct_qa_checkpoint_sha256"
            ),
            "direct_run_summary_sha256": direct_run_artifacts[condition][
                "run_summary_sha256"
            ],
            "direct_run_timing_sha256": direct_run_artifacts[condition][
                "run_timing_sha256"
            ],
            "direct_data_audit_sha256": direct_run_artifacts[condition][
                "data_audit_sha256"
            ],
            "direct_qa_metadata_audit_sha256": direct_run_artifacts[condition][
                "qa_metadata_audit_sha256"
            ],
        }
        for condition, values in artifacts.items()
    }
    for condition, lineage in artifact_lineage.items():
        for field, value in lineage.items():
            if not _is_sha256(value):
                raise ValueError(f"{condition} artifact lineage has invalid {field}.")
    if locked_validation is not None:
        locked_artifacts = _mapping(
            locked_validation.get("artifacts"), label="locked validation artifacts"
        )
        locked_metrics = _mapping(
            locked_validation.get("metrics"), label="locked validation metrics"
        )
        for condition in CONDITIONS:
            locked_lineage = _mapping(
                locked_artifacts.get(condition), label=f"locked artifacts.{condition}"
            )
            for field in (
                "data_audit_sha256",
                "best_checkpoint_sha256",
                "direct_qa_checkpoint_sha256",
                "direct_run_summary_sha256",
                "direct_run_timing_sha256",
                "direct_data_audit_sha256",
                "direct_qa_metadata_audit_sha256",
            ):
                if artifact_lineage[condition][field] != locked_lineage.get(field):
                    raise ValueError(
                        f"{condition} test run does not use the locked validation {field}."
                    )
            resume = _mapping(
                summaries[condition].get("resume"), label=f"{condition} test resume"
            )
            if str(resume.get("sha256", "")).lower() != locked_lineage.get(
                "last_checkpoint_sha256"
            ):
                raise ValueError(
                    f"{condition} test run did not resume the locked validation last checkpoint."
                )
            if json_digest(_metric_payload(summaries[condition], "validation")) != json_digest(
                locked_metrics.get(condition)
            ):
                raise ValueError(
                    f"{condition} validation metrics changed between protocol lock and test."
                )
    try:
        macro_margin = float(evaluation_contract["noninferiority_margin_percentage_points"])
        task_margin = float(evaluation_contract["maximum_per_task_drop_percentage_points"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("Evaluation-contract margins must be numeric.") from error
    if not math.isfinite(macro_margin) or macro_margin < 0.0:
        raise ValueError("The macro non-inferiority margin must be finite and non-negative.")
    if not math.isfinite(task_margin) or task_margin < 0.0:
        raise ValueError("The per-task drop margin must be finite and non-negative.")
    reference = metrics["full_stage1_reference"]
    comparisons: dict[str, Any] = {}
    for condition in CONDITIONS[1:]:
        observed = metrics[condition]
        macro_drop = 100.0 * (
            float(reference["macro_accuracy"]) - float(observed["macro_accuracy"])
        )
        task_drops = {
            task: 100.0
            * (
                float(reference["by_task"][task]["accuracy"])
                - float(observed["by_task"][task]["accuracy"])
            )
            for task in reference["by_task"]
        }
        macro_within = _within_margin(macro_drop, macro_margin)
        task_within = all(_within_margin(value, task_margin) for value in task_drops.values())
        comparisons[condition] = {
            "macro_drop_percentage_points": macro_drop,
            "maximum_task_drop_percentage_points": max(task_drops.values(), default=0.0),
            "task_drop_percentage_points": task_drops,
            "macro_noninferior": macro_within,
            "all_tasks_within_guardrail": task_within,
            "passes_preregistered_screen": macro_within and task_within,
        }

    return {
        "format": RESULT_FORMAT,
        "valid_matched_comparison": True,
        "metric_split": split,
        "conditions": list(CONDITIONS),
        "source_stage1_sha256": contracts["full_stage1_reference"]["source_stage1_sha256"],
        "source_commit": _mapping(
            _mapping(
                contracts["full_stage1_reference"]["traceability"], label="traceability"
            )["source_tree"],
            label="source_tree",
        ).get("commit"),
        "evaluation_contract": evaluation_contract,
        "artifacts": artifact_lineage,
        "metrics": metrics,
        "comparisons_to_reference": comparisons,
        "interpretation": (
            "A passing validation screen supports proceeding to the sealed test comparison; "
            "it is not by itself a final claim that all forms of Stage 1 are unnecessary."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--adapter-only", required=True)
    parser.add_argument("--no-learned-stage1", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = {
        "full_stage1_reference": load_run_artifacts(args.reference),
        "adapter_only": load_run_artifacts(args.adapter_only),
        "no_learned_stage1": load_run_artifacts(args.no_learned_stage1),
    }
    result = compare_runs(artifacts, split=args.split)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        f"valid_matched_comparison=1 split={args.split} output={output} "
        f"sha256={output_sha256} "
        + " ".join(
            f"{condition}_pass={int(values['passes_preregistered_screen'])}"
            for condition, values in result["comparisons_to_reference"].items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
