from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    MATCHED_GROUP_FORMAT,
    PATCH_MATCHED_QA_FORMAT,
    PATCH_QA_BUILD_MARKER,
    PATCH_QA_PROMPT_CONTRACT,
)
from tensor_compression.downstream.patch_qa_prompt import build_prompt  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
    set_default,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as transformers_logging
except ImportError as exc:  # pragma: no cover - dependency error on the execution host
    raise ImportError(
        "scripts/evaluate_frozen_qwen_patch_qa.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


BASELINE_NAME = "frozen_qwen_text_only_no_tensor"
RESULT_FORMAT = "frozen_qwen_patch_qa_baseline_v1"
EXPECTED_TASKS = frozenset(
    {
        "extreme_quadrant",
        "normalized_point_value",
        "point_compare",
        "raw_point_value_with_stats",
        "region_mean_compare",
    }
)
ALLOWED_LABELS = frozenset({"A", "B", "C", "D"})


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return int(dist.get_rank()) if distributed_is_initialized() else 0


def distributed_world_size() -> int:
    return int(dist.get_world_size()) if distributed_is_initialized() else 1


def is_main_process() -> bool:
    return distributed_rank() == 0


def initialize_device(requested: str, timeout_seconds: float) -> torch.device:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("--distributed-timeout-seconds must be finite and positive.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed frozen-Qwen evaluation requires CUDA and NCCL.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible devices."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout),
        )
        return torch.device("cuda", local_rank)
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


def run_on_rank_zero_and_broadcast(operation, stage: str) -> Any:
    if not distributed_is_initialized():
        return operation()
    original_error: BaseException | None = None
    envelope: dict[str, Any] | None = None
    if is_main_process():
        try:
            envelope = {"ok": True, "value": operation()}
        except BaseException as exc:
            original_error = exc
            envelope = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:4000],
            }
    payload = [envelope]
    dist.broadcast_object_list(payload, src=0)
    received = payload[0]
    if not isinstance(received, Mapping) or not bool(received.get("ok", False)):
        error_type = (
            str(received.get("error_type", "RuntimeError"))
            if isinstance(received, Mapping)
            else "RuntimeError"
        )
        error_message = (
            str(received.get("error_message", "rank 0 returned an invalid result"))
            if isinstance(received, Mapping)
            else "rank 0 returned an invalid result"
        )
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"Rank-0 {stage} failed with {error_type}: {error_message}")
    return received.get("value")


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_dump_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(output_root: str | Path, run_name: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{timestamp}_{run_name}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{run_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def build_distributed_run_dir(output_root: str | Path, run_name: str) -> Path:
    path = run_on_rank_zero_and_broadcast(
        lambda: str(create_run_dir(output_root, run_name)),
        "run directory creation",
    )
    if not isinstance(path, str) or not path:
        raise RuntimeError("Rank 0 broadcast an invalid output directory.")
    return Path(path)


def parse_splits(raw: str) -> list[str]:
    splits = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split name.")
    if len(set(splits)) != len(splits):
        raise ValueError(f"--splits contains duplicates: {splits}.")
    invalid = [split for split in splits if not split.replace("_", "").isalnum()]
    if invalid:
        raise ValueError(f"Invalid split names: {invalid}.")
    return splits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completely frozen Qwen on Stage-2B patch QA using question text only. "
            "No tensor adapter, tensor latent, or adapter checkpoint is constructed or loaded."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None, help="Comma-separated QA splits.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Deterministic prefix cap per split for smoke tests; omit for formal full-split evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-rank evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--prompt-template", choices=("task_specific",), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--serialize-llm-loading", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-host-memory-available-gib", type=float, default=None)
    parser.add_argument("--distributed-timeout-seconds", type=float, default=None)
    parser.add_argument("--require-formal-contract", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return apply_config_defaults(parser.parse_args(argv))


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_mapping(args.config)
    model_local_dir = first_nested(config, ["model.local_dir"])
    model_name = first_nested(config, ["model.name_or_path", "model.model_name_or_path"])
    if args.model_name_or_path is None:
        args.model_name_or_path = (
            resolve_path_string(model_local_dir, PROJECT_ROOT) if model_local_dir else model_name
        )

    path_defaults = {
        "qa_dir": first_nested(config, ["patch_qa.stage2b_qa_dir"]),
        "cache_dir": first_nested(config, ["model.cache_dir", "storage.hf_home"]),
        "hf_home": first_nested(config, ["storage.hf_home"]),
        "output_root": first_nested(config, ["llm_training.output_root", "storage.output_root"]),
    }
    for attr, value in path_defaults.items():
        if getattr(args, attr) is None and value is not None:
            setattr(args, attr, resolve_path_string(value, PROJECT_ROOT))

    val_split = str(first_nested(config, ["llm_training.val_split"], "val"))
    test_split = str(first_nested(config, ["llm_training.test_split"], "test"))
    set_default(args, "run_name", None, BASELINE_NAME)
    set_default(args, "splits", None, f"{val_split},{test_split}")
    set_default(args, "batch_size", first_nested(config, ["llm_training.eval_batch_size"]), 8)
    set_default(args, "num_workers", first_nested(config, ["llm_training.num_workers"]), 0)
    set_default(
        args,
        "max_prompt_tokens",
        first_nested(config, ["llm_training.max_prompt_tokens"]),
        512,
    )
    set_default(
        args,
        "prompt_template",
        first_nested(config, ["llm_training.prompt_template"]),
        "task_specific",
    )
    set_default(args, "device", first_nested(config, ["llm_training.device", "runtime.device"]), "auto")
    set_default(
        args,
        "torch_dtype",
        first_nested(config, ["llm_training.torch_dtype", "model.torch_dtype"]),
        "bfloat16",
    )
    set_default(args, "trust_remote_code", first_nested(config, ["model.trust_remote_code"]), False)
    set_default(
        args,
        "low_cpu_mem_usage",
        first_nested(config, ["llm_training.low_cpu_mem_usage", "model.low_cpu_mem_usage"]),
        True,
    )
    set_default(
        args,
        "serialize_llm_loading",
        first_nested(config, ["llm_training.serialize_llm_loading"]),
        True,
    )
    set_default(
        args,
        "min_host_memory_available_gib",
        first_nested(config, ["llm_training.min_host_memory_available_gib"]),
        0.0,
    )
    set_default(
        args,
        "distributed_timeout_seconds",
        first_nested(config, ["llm_training.distributed_timeout_seconds"]),
        1800.0,
    )
    set_default(args, "require_formal_contract", None, True)
    set_default(args, "console_progress", first_nested(config, ["llm_training.console_progress"]), False)
    set_default(args, "seed", first_nested(config, ["runtime.seed", "llm_training.shuffle_seed"]), 42)

    missing = [
        name
        for name in ("qa_dir", "model_name_or_path", "output_root")
        if getattr(args, name, None) in {None, ""}
    ]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Missing required argument(s): {flags}.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative.")
    if int(args.max_prompt_tokens) <= 0:
        raise ValueError("--max-prompt-tokens must be positive.")
    if args.max_records is not None and int(args.max_records) <= 0:
        raise ValueError("--max-records must be positive when provided.")
    if float(args.min_host_memory_available_gib) < 0.0:
        raise ValueError("--min-host-memory-available-gib must be non-negative.")
    args.splits = ",".join(parse_splits(args.splits))
    if str(args.prompt_template) != "task_specific":
        raise ValueError("Formal frozen-Qwen baseline requires prompt_template=task_specific.")
    return args


def apply_runtime_environment(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(args.hf_home))
    if args.cache_dir:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))
    if not bool(args.console_progress):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        transformers_logging.disable_progress_bar()


def qa_path(qa_dir: str | Path, split: str) -> Path:
    path = Path(qa_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"QA split file not found: {path}")
    return path


def audit_qa_metadata(
    qa_dir: str | Path,
    splits: Sequence[str],
    require_formal_contract: bool,
) -> dict[str, Any]:
    root = Path(qa_dir)
    build_marker = root / PATCH_QA_BUILD_MARKER
    if build_marker.exists():
        raise RuntimeError(f"Stage-2B QA is marked as incomplete or active: {build_marker}")
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        if require_formal_contract:
            raise FileNotFoundError(f"Formal baseline requires QA metadata: {metadata_path}")
        metadata: Mapping[str, Any] = {}
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Expected a JSON object in {metadata_path}.")
        metadata = loaded

    observed = {
        "format": str(metadata.get("format", "")),
        "matched_group_format": str(metadata.get("matched_group_format", "")),
        "prompt_contract": str(metadata.get("prompt_contract", "")),
        "natural_language_coordinate_origin": int(
            metadata.get("natural_language_coordinate_origin", -1)
        ),
        "split_mode": str(metadata.get("split_mode", "")),
        "requires_explicit_group_sampler": bool(
            metadata.get("requires_explicit_group_sampler", False)
        ),
    }
    if require_formal_contract:
        expected = {
            "format": PATCH_MATCHED_QA_FORMAT,
            "matched_group_format": MATCHED_GROUP_FORMAT,
            "prompt_contract": PATCH_QA_PROMPT_CONTRACT,
            "natural_language_coordinate_origin": 1,
            "split_mode": "sample",
            "requires_explicit_group_sampler": True,
        }
        mismatches = {
            key: {"expected": value, "observed": observed[key]}
            for key, value in expected.items()
            if observed[key] != value
        }
        if mismatches:
            raise ValueError(f"Formal Stage-2B QA metadata contract mismatch: {mismatches}")

    declared_hashes = metadata.get("output_split_sha256", {})
    if not isinstance(declared_hashes, Mapping):
        declared_hashes = {}
    split_files: dict[str, Any] = {}
    for split in splits:
        path = qa_path(root, split)
        actual_hash = sha256_file(path)
        declared_hash = str(declared_hashes.get(split, ""))
        if require_formal_contract and not declared_hash:
            raise ValueError(f"Formal metadata does not declare output_split_sha256[{split!r}].")
        if declared_hash and actual_hash != declared_hash:
            raise ValueError(
                f"QA split changed after metadata was written: split={split}, "
                f"declared={declared_hash}, actual={actual_hash}."
            )
        split_files[split] = {
            "path": str(path),
            "sha256": actual_hash,
            "matches_declared_sha256": bool(declared_hash and actual_hash == declared_hash),
        }

    summary = metadata.get("summary", {})
    summary_splits = summary.get("splits", {}) if isinstance(summary, Mapping) else {}
    declared_records = {
        split: int(summary_splits.get(split, {}).get("qa_records", -1))
        if isinstance(summary_splits, Mapping)
        and isinstance(summary_splits.get(split), Mapping)
        else -1
        for split in splits
    }
    return {
        "available": bool(metadata_path.exists()),
        "path": str(metadata_path),
        "sha256": sha256_file(metadata_path) if metadata_path.exists() else None,
        **observed,
        "fields": [str(field) for field in metadata.get("fields", [])],
        "declared_records": declared_records,
        "split_files": split_files,
        "formal_contract_required": bool(require_formal_contract),
        "formal_contract_passed": bool(require_formal_contract),
        "stage1_checkpoint_opened": False,
        "latent_contract_evaluated": False,
    }


def load_qa_records(path: str | Path, max_records: int | None = None) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    source_oracle_records = 0
    limit = None if max_records is None else int(max_records)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(records) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            source_oracle_records += int("oracle" in payload)
            payload.pop("oracle", None)
            records.append(payload)
    return records, source_oracle_records


class FrozenQwenQADataset(Dataset):
    def __init__(self, path: str | Path, max_records: int | None = None) -> None:
        self.path = Path(path)
        self.records, self.source_oracle_records = load_qa_records(self.path, max_records)
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.path}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"index": int(index), "record": self.records[index]}


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation records without padding or repeating any example."""

    def __init__(self, dataset: Dataset, rank: int, num_replicas: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.num_replicas = max(1, int(num_replicas))
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.num_replicas) + 1


def collate_records(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "indices": [int(item["index"]) for item in items],
        "records": [item["record"] for item in items],
    }


def prompt_only_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only fields consumed by the shared prompt contract."""

    return {
        "task_type": str(record.get("task_type", "")),
        "query": record.get("query"),
        "question": record.get("question"),
        "choices": list(record.get("choices", [])),
    }


def render_prompt(record: Mapping[str, Any], prompt_template: str) -> str:
    return build_prompt(prompt_only_record(record), prompt_template=prompt_template)


def record_field(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    metadata_field = metadata.get("field") if isinstance(metadata, Mapping) else None
    return str(record.get("field") or metadata_field or "unknown")


def audit_qa_records(
    datasets: Mapping[str, FrozenQwenQADataset],
    metadata_audit: Mapping[str, Any],
    require_formal_contract: bool,
    complete_splits: bool,
) -> dict[str, Any]:
    split_samples: dict[str, set[int]] = {}
    split_states: dict[str, set[str]] = {}
    split_qa_ids: dict[str, set[str]] = {}
    output: dict[str, Any] = {}
    metadata_fields = set(str(field) for field in metadata_audit.get("fields", []))
    for split, dataset in datasets.items():
        qa_ids: set[str] = set()
        states: set[str] = set()
        samples: set[int] = set()
        tasks: Counter[str] = Counter()
        fields: Counter[str] = Counter()
        answers_by_task: dict[str, Counter[str]] = defaultdict(Counter)
        choices_by_task: dict[str, set[str]] = defaultdict(set)
        candidate_counts: Counter[int] = Counter()
        for record in dataset.records:
            qa_id = str(record.get("qa_id", ""))
            if not qa_id or qa_id in qa_ids:
                raise ValueError(f"Missing or duplicate qa_id in {split}: {qa_id!r}")
            qa_ids.add(qa_id)
            state_ref = str(record.get("state_ref", ""))
            if not state_ref:
                raise ValueError(f"Record {qa_id} has no state_ref.")
            states.add(state_ref)
            if "sample_index" not in record:
                raise ValueError(f"Record {qa_id} has no sample_index.")
            samples.add(int(record["sample_index"]))

            task = str(record.get("task_type", ""))
            field = record_field(record)
            choices = record.get("choices")
            answer = str(record.get("answer", ""))
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                raise ValueError(f"Record {qa_id} has invalid choices.")
            labels = [str(choice) for choice in choices]
            if len(labels) not in {2, 4} or len(set(labels)) != len(labels):
                raise ValueError(f"Record {qa_id} must have two or four unique choices, got {labels}.")
            if not set(labels).issubset(ALLOWED_LABELS) or answer not in labels:
                raise ValueError(f"Record {qa_id} has invalid labels/answer: choices={labels}, answer={answer}.")
            query = str(record.get("query") or record.get("question") or "")
            if not query:
                raise ValueError(f"Record {qa_id} has an empty query.")

            prompt = render_prompt(record, "task_specific")
            if prompt != build_prompt(record, prompt_template="task_specific"):
                raise RuntimeError(f"Prompt-only projection changed the formal prompt for {qa_id}.")
            mutated = dict(record)
            mutated["answer"] = "__FORBIDDEN_ANSWER_SENTINEL__"
            mutated["oracle"] = "__FORBIDDEN_ORACLE_SENTINEL__"
            if render_prompt(mutated, "task_specific") != prompt:
                raise RuntimeError(f"Answer/oracle changed the model prompt for {qa_id}.")

            tasks[task] += 1
            fields[field] += 1
            answers_by_task[task][answer] += 1
            choices_by_task[task].update(labels)
            candidate_counts[len(labels)] += 1

        observed_tasks = set(tasks)
        observed_fields = set(fields)
        if require_formal_contract and complete_splits:
            if observed_tasks != EXPECTED_TASKS:
                raise ValueError(
                    f"Formal split {split} task mismatch: expected={sorted(EXPECTED_TASKS)}, "
                    f"observed={sorted(observed_tasks)}."
                )
            if metadata_fields and observed_fields != metadata_fields:
                raise ValueError(
                    f"Formal split {split} field mismatch: metadata={sorted(metadata_fields)}, "
                    f"observed={sorted(observed_fields)}."
                )
            missing_answers = {
                task: sorted(choices_by_task[task] - set(answers_by_task[task]))
                for task in choices_by_task
                if choices_by_task[task] - set(answers_by_task[task])
            }
            if missing_answers:
                raise ValueError(f"Formal split {split} lacks answer-label coverage: {missing_answers}")
            declared = int(metadata_audit.get("declared_records", {}).get(split, -1))
            if declared >= 0 and len(dataset) != declared:
                raise ValueError(
                    f"Formal split {split} record count differs from metadata: "
                    f"loaded={len(dataset)}, declared={declared}."
                )

        split_samples[split] = samples
        split_states[split] = states
        split_qa_ids[split] = qa_ids
        output[split] = {
            "records": len(dataset),
            "states": len(states),
            "samples": len(samples),
            "by_task": dict(sorted(tasks.items())),
            "by_field": dict(sorted(fields.items())),
            "answers_by_task": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(answers_by_task.items())
            },
            "candidate_count_distribution": {
                str(count): total for count, total in sorted(candidate_counts.items())
            },
            "oracle_fields_removed": int(dataset.source_oracle_records),
        }

    overlaps: dict[str, Any] = {}
    split_names = list(datasets)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}_{right}"
            overlap = {
                "qa_ids": len(split_qa_ids[left] & split_qa_ids[right]),
                "states": len(split_states[left] & split_states[right]),
                "samples": len(split_samples[left] & split_samples[right]),
            }
            overlaps[key] = overlap
            if require_formal_contract and any(overlap.values()):
                raise ValueError(f"Formal evaluation splits overlap: {key}={overlap}")
    return {
        "splits": output,
        "overlaps": overlaps,
        "complete_split_contract_checked": bool(complete_splits),
        "prompt_projection_excludes": [
            "answer",
            "oracle",
            "latent_ref",
            "latent_path",
            "grounding_target",
            "matched_group",
            "prompt_data",
        ],
    }


def audit_prompt_tokenization(
    datasets: Mapping[str, FrozenQwenQADataset],
    tokenizer,
    max_prompt_tokens: int,
    prompt_template: str,
) -> dict[str, Any]:
    limit = int(max_prompt_tokens)
    split_summary: dict[str, Any] = {}
    all_labels: set[str] = set()
    for split, dataset in datasets.items():
        total_tokens = 0
        max_tokens = 0
        max_record: str | None = None
        by_task: dict[str, dict[str, int]] = defaultdict(
            lambda: {"records": 0, "total_tokens": 0, "max_tokens": 0}
        )
        for record in dataset.records:
            prompt = render_prompt(record, prompt_template)
            token_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            token_count = len(token_ids)
            if token_count > limit:
                raise ValueError(
                    f"Prompt {record.get('qa_id')} uses {token_count} tokens, exceeding "
                    f"max_prompt_tokens={limit}; formal evaluation never truncates prompts."
                )
            total_tokens += token_count
            if token_count > max_tokens:
                max_tokens = token_count
                max_record = str(record.get("qa_id", ""))
            task = str(record.get("task_type", "unknown"))
            by_task[task]["records"] += 1
            by_task[task]["total_tokens"] += token_count
            by_task[task]["max_tokens"] = max(by_task[task]["max_tokens"], token_count)
            all_labels.update(str(choice) for choice in record["choices"])
        split_summary[split] = {
            "records": len(dataset),
            "mean_tokens": total_tokens / len(dataset),
            "max_tokens": max_tokens,
            "max_token_record": max_record,
            "truncated_records": 0,
            "by_task": {
                task: {
                    "records": values["records"],
                    "mean_tokens": values["total_tokens"] / max(1, values["records"]),
                    "max_tokens": values["max_tokens"],
                }
                for task, values in sorted(by_task.items())
            },
        }

    label_token_ids: dict[str, int] = {}
    for label in sorted(all_labels):
        encoded = tokenizer(
            " " + label,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                f"Formal restricted-label scoring requires one token for label {label!r}; got {encoded}."
            )
        label_token_ids[label] = int(encoded[0])
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise ValueError(f"Choice labels do not have unique token ids: {label_token_ids}")
    return {
        "prompt_template": prompt_template,
        "max_prompt_tokens": limit,
        "all_prompts_fit": True,
        "tokenizer_padding_side": str(tokenizer.padding_side),
        "label_encoding": "tokenizer(' ' + label, add_special_tokens=False)",
        "label_token_ids": label_token_ids,
        "all_labels_single_token_and_unique": True,
        "splits": split_summary,
    }


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_model_dtype(raw: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if raw == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[raw]


def host_memory_available_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                parts = line.partition(":")[2].strip().split()
                return float(parts[0]) / 1024**2 if parts else None
    return None


def enforce_host_memory_floor(minimum_gib: float, stage: str) -> None:
    local = host_memory_available_gib()
    gathered: list[float | None]
    if distributed_is_initialized():
        gathered = [None] * distributed_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    available = [float(value) for value in gathered if value is not None]
    if available and float(minimum_gib) > 0.0 and min(available) < float(minimum_gib):
        raise RuntimeError(
            f"Host memory guard stopped {stage}: MemAvailable={min(available):.2f} GiB is below "
            f"the configured floor {float(minimum_gib):.2f} GiB."
        )


def load_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token for batched evaluation.")
    return tokenizer


def load_frozen_qwen(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, torch.dtype]:
    dtype = resolve_model_dtype(str(args.torch_dtype), device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
    )
    model.to(device)
    disable_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_type = str(getattr(model.config, "model_type", ""))
    if "qwen" not in model_type.casefold():
        raise ValueError(
            f"This baseline is restricted to Qwen causal LMs; loaded model_type={model_type!r}."
        )
    return model, dtype


def load_frozen_qwen_serialized(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, torch.dtype]:
    if not distributed_is_initialized() or not bool(args.serialize_llm_loading):
        enforce_host_memory_floor(float(args.min_host_memory_available_gib), "Qwen loading")
        return load_frozen_qwen(args, device)

    local_model: nn.Module | None = None
    local_dtype: torch.dtype | None = None
    for load_rank in range(distributed_world_size()):
        enforce_host_memory_floor(
            float(args.min_host_memory_available_gib),
            f"Qwen loading for rank {load_rank}",
        )
        local_error: BaseException | None = None
        if distributed_rank() == load_rank:
            try:
                print(
                    f"startup=qwen_load rank={load_rank}/{distributed_world_size() - 1}",
                    flush=True,
                )
                local_model, local_dtype = load_frozen_qwen(args, device)
            except BaseException as exc:
                local_error = exc
        error_payload = [
            None
            if local_error is None
            else f"{type(local_error).__name__}: {str(local_error)[:2000]}"
        ]
        dist.broadcast_object_list(error_payload, src=load_rank)
        if error_payload[0] is not None:
            raise RuntimeError(
                f"Distributed Qwen loading failed on rank {load_rank}: {error_payload[0]}"
            ) from local_error
        distributed_barrier()
    if local_model is None or local_dtype is None:
        raise RuntimeError("The local rank did not construct its frozen Qwen replica.")
    return local_model, local_dtype


def decoder_for_causal_lm(model: nn.Module) -> nn.Module:
    get_decoder = getattr(model, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    if decoder is None or decoder is model:
        prefix = str(getattr(model, "base_model_prefix", ""))
        decoder = getattr(model, prefix, None) if prefix else None
    if decoder is None or decoder is model or not isinstance(decoder, nn.Module):
        raise ValueError("The Qwen causal LM does not expose its decoder backbone.")
    return decoder


def audit_frozen_qwen(model: nn.Module, dtype: torch.dtype) -> dict[str, Any]:
    parameters = list(model.parameters())
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    training_modules = [name for name, module in model.named_modules() if module.training]
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    if trainable != 0 or model.training or training_modules or checkpointing:
        raise RuntimeError(
            "Frozen-Qwen contract failed: "
            f"trainable={trainable}, model.training={model.training}, "
            f"training_modules={training_modules[:8]}, gradient_checkpointing={checkpointing}."
        )
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Qwen does not expose output embeddings for restricted-label scoring.")
    decoder_for_causal_lm(model)
    return {
        "class": type(model).__name__,
        "model_type": str(getattr(model.config, "model_type", "")),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": trainable,
        "training_module_count": len(training_modules),
        "eval_mode": not bool(model.training),
        "gradient_checkpointing": checkpointing,
        "config_use_cache": bool(getattr(model.config, "use_cache", False)),
        "execution_dtype": str(dtype).replace("torch.", ""),
    }


def last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected [batch,tokens] attention mask, got {tuple(attention_mask.shape)}.")
    valid = attention_mask.to(dtype=torch.bool)
    if int(valid.shape[1]) == 0 or not bool(valid.any(dim=1).all()):
        raise ValueError("Every prompt must contain at least one non-padding token.")
    positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0).expand_as(valid)
    return positions.masked_fill(~valid, -1).max(dim=1).values


@torch.inference_mode()
def score_prompt_batch(
    model: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    choices_by_record: Sequence[Sequence[str]],
    label_token_ids: Mapping[str, int],
    device: torch.device,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    if torch.is_grad_enabled():
        raise RuntimeError("Frozen-Qwen scoring unexpectedly enabled autograd.")
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    attention_mask = encoded["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1)
    if int(prompt_lengths.max().item()) > int(max_prompt_tokens):
        raise ValueError(
            f"A prompt exceeds max_prompt_tokens={int(max_prompt_tokens)}; prompts are never truncated."
        )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = attention_mask.to(device)
    decoder = decoder_for_causal_lm(model)
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    last_indices = last_nonpadding_indices(attention_mask)
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    last_hidden = outputs.last_hidden_state[batch_indices, last_indices]
    logits = model.get_output_embeddings()(last_hidden).float()

    scored: list[dict[str, Any]] = []
    for row, raw_choices in enumerate(choices_by_record):
        choices = [str(choice) for choice in raw_choices]
        candidate_ids = torch.tensor(
            [int(label_token_ids[choice]) for choice in choices],
            dtype=torch.long,
            device=device,
        )
        candidate_logits = logits[row].index_select(0, candidate_ids)
        probabilities = F.softmax(candidate_logits, dim=0)
        prediction_index = int(torch.argmax(candidate_logits).item())
        scored.append(
            {
                "prediction": choices[prediction_index],
                "probabilities": {
                    choice: float(probabilities[index].item())
                    for index, choice in enumerate(choices)
                },
                "prompt_tokens": int(prompt_lengths[row].item()),
            }
        )
    return scored


def empty_metric_payload() -> dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "restricted_nll_sum": 0.0,
        "target_probability_sum": 0.0,
        "prediction_confidence_sum": 0.0,
        "uniform_chance_sum": 0.0,
        "prompt_token_sum": 0,
        "prompt_token_max": 0,
        "task_total": defaultdict(int),
        "task_correct": defaultdict(int),
        "field_total": defaultdict(int),
        "field_correct": defaultdict(int),
        "task_field_total": defaultdict(int),
        "task_field_correct": defaultdict(int),
        "answer_label_total": defaultdict(int),
        "prediction_label_total": defaultdict(int),
        "task_prediction_label_total": defaultdict(int),
        "confusion_total": defaultdict(int),
        "candidate_count_total": defaultdict(int),
        "indices": [],
    }


def update_metric_payload(
    payload: dict[str, Any],
    record: Mapping[str, Any],
    scored: Mapping[str, Any],
    index: int,
) -> None:
    answer = str(record["answer"])
    prediction = str(scored["prediction"])
    probabilities = scored["probabilities"]
    if not isinstance(probabilities, Mapping) or answer not in probabilities:
        raise ValueError(f"Scored probabilities omit answer {answer!r}.")
    target_probability = float(probabilities[answer])
    prediction_probability = float(probabilities[prediction])
    hit = int(prediction == answer)
    task = str(record.get("task_type", "unknown"))
    field = record_field(record)
    task_field = f"{task}/{field}"
    choices = [str(choice) for choice in record["choices"]]

    payload["total"] += 1
    payload["correct"] += hit
    payload["restricted_nll_sum"] += -math.log(max(target_probability, 1.0e-30))
    payload["target_probability_sum"] += target_probability
    payload["prediction_confidence_sum"] += prediction_probability
    payload["uniform_chance_sum"] += 1.0 / len(choices)
    payload["prompt_token_sum"] += int(scored["prompt_tokens"])
    payload["prompt_token_max"] = max(payload["prompt_token_max"], int(scored["prompt_tokens"]))
    payload["task_total"][task] += 1
    payload["task_correct"][task] += hit
    payload["field_total"][field] += 1
    payload["field_correct"][field] += hit
    payload["task_field_total"][task_field] += 1
    payload["task_field_correct"][task_field] += hit
    payload["answer_label_total"][answer] += 1
    payload["prediction_label_total"][prediction] += 1
    payload["task_prediction_label_total"][f"{task}/{prediction}"] += 1
    payload["confusion_total"][f"{answer}/{prediction}"] += 1
    payload["candidate_count_total"][str(len(choices))] += 1
    payload["indices"].append(int(index))


def serializable_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: dict(value) if isinstance(value, defaultdict) else list(value) if key == "indices" else value
        for key, value in payload.items()
    }


def merge_metric_payloads(
    payloads: Sequence[Mapping[str, Any]],
    expected_total: int,
) -> dict[str, Any]:
    scalar_ints = ("total", "correct", "prompt_token_sum")
    scalar_floats = (
        "restricted_nll_sum",
        "target_probability_sum",
        "prediction_confidence_sum",
        "uniform_chance_sum",
    )
    map_names = (
        "task_total",
        "task_correct",
        "field_total",
        "field_correct",
        "task_field_total",
        "task_field_correct",
        "answer_label_total",
        "prediction_label_total",
        "task_prediction_label_total",
        "confusion_total",
        "candidate_count_total",
    )
    merged: dict[str, Any] = {name: 0 for name in scalar_ints}
    merged.update({name: 0.0 for name in scalar_floats})
    merged["prompt_token_max"] = 0
    for name in map_names:
        merged[name] = defaultdict(int)
    indices: list[int] = []
    for payload in payloads:
        for name in scalar_ints:
            merged[name] += int(payload.get(name, 0))
        for name in scalar_floats:
            merged[name] += float(payload.get(name, 0.0))
        merged["prompt_token_max"] = max(
            int(merged["prompt_token_max"]), int(payload.get("prompt_token_max", 0))
        )
        for name in map_names:
            values = payload.get(name, {})
            if not isinstance(values, Mapping):
                raise TypeError(f"Metric payload {name} must be a mapping.")
            for key, value in values.items():
                merged[name][str(key)] += int(value)
        indices.extend(int(value) for value in payload.get("indices", []))

    expected_indices = list(range(int(expected_total)))
    if sorted(indices) != expected_indices:
        duplicates = len(indices) - len(set(indices))
        missing = sorted(set(expected_indices) - set(indices))[:16]
        extras = sorted(set(indices) - set(expected_indices))[:16]
        raise RuntimeError(
            "Distributed evaluation shard audit failed: "
            f"records={len(indices)}, expected={expected_total}, duplicates={duplicates}, "
            f"missing={missing}, extras={extras}."
        )
    if int(merged["total"]) != int(expected_total):
        raise RuntimeError(
            f"Metric total {merged['total']} does not equal dataset size {expected_total}."
        )
    merged["indices_sha256"] = hashlib.sha256(
        ",".join(str(index) for index in sorted(indices)).encode("ascii")
    ).hexdigest()
    merged["world_shard_record_counts"] = [int(payload.get("total", 0)) for payload in payloads]
    return merged


def grouped_accuracy(
    totals: Mapping[str, int],
    correct: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "accuracy": int(correct.get(key, 0)) / max(1, int(total)),
            "correct": int(correct.get(key, 0)),
            "total": int(total),
        }
        for key, total in sorted(totals.items())
    }


def nested_distribution(flat: Mapping[str, int]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for key, value in sorted(flat.items()):
        outer, separator, inner = str(key).rpartition("/")
        if not separator:
            outer, inner = "unknown", str(key)
        nested[outer][inner] = int(value)
    return dict(nested)


def finalize_metrics(merged: Mapping[str, Any]) -> dict[str, Any]:
    total = int(merged["total"])
    by_task = grouped_accuracy(merged["task_total"], merged["task_correct"])
    return {
        "accuracy": int(merged["correct"]) / max(1, total),
        "correct": int(merged["correct"]),
        "total": total,
        "macro_task_accuracy": sum(item["accuracy"] for item in by_task.values())
        / max(1, len(by_task)),
        "uniform_random_expected_accuracy": float(merged["uniform_chance_sum"]) / max(1, total),
        "mean_restricted_nll": float(merged["restricted_nll_sum"]) / max(1, total),
        "mean_target_probability": float(merged["target_probability_sum"]) / max(1, total),
        "mean_prediction_confidence": float(merged["prediction_confidence_sum"]) / max(1, total),
        "prompt_tokens": {
            "mean": int(merged["prompt_token_sum"]) / max(1, total),
            "max": int(merged["prompt_token_max"]),
        },
        "by_task": by_task,
        "by_field": grouped_accuracy(merged["field_total"], merged["field_correct"]),
        "by_task_field": grouped_accuracy(
            merged["task_field_total"], merged["task_field_correct"]
        ),
        "answer_label_distribution": dict(sorted(merged["answer_label_total"].items())),
        "prediction_label_distribution": dict(
            sorted(merged["prediction_label_total"].items())
        ),
        "prediction_label_distribution_by_task": nested_distribution(
            merged["task_prediction_label_total"]
        ),
        "target_prediction_confusion": nested_distribution(merged["confusion_total"]),
        "candidate_count_distribution": dict(sorted(merged["candidate_count_total"].items())),
        "distributed_shard_audit": {
            "exact_no_padding_no_repeat": True,
            "indices_sha256": str(merged["indices_sha256"]),
            "records_by_rank": list(merged["world_shard_record_counts"]),
        },
    }


@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    tokenizer,
    dataset: FrozenQwenQADataset,
    label_token_ids: Mapping[str, int],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
) -> dict[str, Any]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen must remain frozen and in eval mode throughout evaluation.")
    sampler = (
        ExactDistributedEvalSampler(
            dataset,
            rank=distributed_rank(),
            num_replicas=distributed_world_size(),
        )
        if distributed_is_initialized()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(args.num_workers),
        persistent_workers=False,
        prefetch_factor=1 if int(args.num_workers) > 0 else None,
        pin_memory=device.type == "cuda",
        collate_fn=collate_records,
    )
    local = empty_metric_payload()
    iterator = tqdm(
        loader,
        desc=f"Frozen Qwen [{split}] rank {distributed_rank()}",
        leave=False,
        disable=not bool(args.console_progress) or not is_main_process(),
    )
    for batch in iterator:
        records = batch["records"]
        prompts = [render_prompt(record, str(args.prompt_template)) for record in records]
        choices_by_record = [[str(choice) for choice in record["choices"]] for record in records]
        scored = score_prompt_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            choices_by_record=choices_by_record,
            label_token_ids=label_token_ids,
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
        )
        for index, record, result in zip(batch["indices"], records, scored):
            update_metric_payload(local, record, result, index)

    local_serializable = serializable_metric_payload(local)
    if distributed_is_initialized():
        gathered: list[Mapping[str, Any] | None] = [None] * distributed_world_size()
        dist.all_gather_object(gathered, local_serializable)
        payloads = [payload for payload in gathered if payload is not None]
    else:
        payloads = [local_serializable]
    merged = merge_metric_payloads(payloads, expected_total=len(dataset))
    return finalize_metrics(merged)


def redact_config(value: Any, key: str = "") -> Any:
    lowered = key.casefold()
    if any(term in lowered for term in ("api_key", "password", "secret", "access_token")):
        return "<redacted>" if value is not None and value != "" else value
    if isinstance(value, Mapping):
        return {str(item_key): redact_config(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def runtime_contract() -> dict[str, Any]:
    return {
        "baseline_name": BASELINE_NAME,
        "model": "Qwen causal LM only",
        "model_parameters_frozen": True,
        "model_eval_mode": True,
        "autograd_during_forward": False,
        "optimizer_created": False,
        "backward_called": False,
        "adapter_instantiated": False,
        "adapter_checkpoint_loaded": False,
        "stage1_checkpoint_loaded": False,
        "latent_files_opened": False,
        "tensor_serialized_into_text": False,
        "soft_prefix_tokens": 0,
        "forward_model_inputs": ["input_ids", "attention_mask"],
        "prompt_record_fields": ["task_type", "query", "question", "choices"],
        "answer_used_by_model_forward": False,
        "answer_used_after_forward_for_metrics_only": True,
        "scoring": "next-token logits restricted to each record's displayed choice labels",
        "interpretation": (
            "No-tensor language/label-prior lower baseline; it is not information-equivalent to the "
            "tensor-adapter model."
        ),
    }


def print_split_metrics(split: str, metrics: Mapping[str, Any]) -> None:
    task_text = ", ".join(
        f"{task}={float(values['accuracy']):.4f}"
        for task, values in sorted(metrics.get("by_task", {}).items())
    )
    print(
        f"split={split} accuracy={float(metrics['accuracy']):.4f} "
        f"correct={int(metrics['correct'])}/{int(metrics['total'])} tasks[{task_text}]",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_runtime_environment(args)
    device: torch.device | None = None
    started = time.monotonic()
    try:
        device = initialize_device(str(args.device), float(args.distributed_timeout_seconds))
        seed_everything(int(args.seed) + distributed_rank())
        splits = parse_splits(args.splits)
        run_dir = build_distributed_run_dir(args.output_root, args.run_name)

        raw_config = load_yaml_mapping(args.config)
        if is_main_process():
            atomic_dump_json(
                run_dir / "resolved_run_config.json",
                {
                    "format": RESULT_FORMAT,
                    "created_at": local_timestamp(),
                    "source_config": str(args.config or ""),
                    "source_config_sha256": sha256_file(args.config) if args.config else None,
                    "resolved_args": vars(args),
                    "config_snapshot": redact_config(raw_config),
                },
            )

        metadata_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_qa_metadata(
                args.qa_dir,
                splits,
                require_formal_contract=bool(args.require_formal_contract),
            ),
            "QA metadata audit",
        )
        datasets = {
            split: FrozenQwenQADataset(
                qa_path(args.qa_dir, split),
                max_records=args.max_records,
            )
            for split in splits
        }
        qa_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_qa_records(
                datasets,
                metadata_audit=metadata_audit,
                require_formal_contract=bool(args.require_formal_contract),
                complete_splits=args.max_records is None,
            ),
            "QA record audit",
        )
        if is_main_process():
            atomic_dump_json(
                run_dir / "qa_only_audit.json",
                {"metadata": metadata_audit, "records": qa_audit},
            )

        tokenizer = load_tokenizer(args)
        prompt_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_prompt_tokenization(
                datasets,
                tokenizer=tokenizer,
                max_prompt_tokens=int(args.max_prompt_tokens),
                prompt_template=str(args.prompt_template),
            ),
            "prompt/tokenizer audit",
        )
        label_token_ids = {
            str(label): int(token_id)
            for label, token_id in prompt_audit["label_token_ids"].items()
        }

        if is_main_process():
            print(
                f"baseline={BASELINE_NAME} model={args.model_name_or_path} "
                f"splits={','.join(splits)} batch_per_rank={int(args.batch_size)} "
                f"world_size={distributed_world_size()} output={run_dir}",
                flush=True,
            )
        model, dtype = load_frozen_qwen_serialized(args, device)
        model_audit = audit_frozen_qwen(model, dtype)

        split_metrics: dict[str, Any] = {}
        for split in splits:
            distributed_barrier()
            split_started = time.monotonic()
            metrics = evaluate_split(
                model=model,
                tokenizer=tokenizer,
                dataset=datasets[split],
                label_token_ids=label_token_ids,
                device=device,
                args=args,
                split=split,
            )
            metrics["elapsed_seconds"] = time.monotonic() - split_started
            split_metrics[split] = metrics
            if is_main_process():
                print_split_metrics(split, metrics)

        result = {
            "format": RESULT_FORMAT,
            "baseline_name": BASELINE_NAME,
            "completed_at": local_timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "world_size": distributed_world_size(),
            "device_type": device.type,
            "model_name_or_path": str(args.model_name_or_path),
            "model_audit": model_audit,
            "runtime_contract": runtime_contract(),
            "qa_metadata_audit": metadata_audit,
            "qa_record_audit": qa_audit,
            "prompt_tokenization_audit": prompt_audit,
            "splits": split_metrics,
        }
        if is_main_process():
            atomic_dump_json(run_dir / "frozen_qwen_results.json", result)
            print(f"results={run_dir / 'frozen_qwen_results.json'}", flush=True)
        distributed_barrier()
    finally:
        if distributed_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
