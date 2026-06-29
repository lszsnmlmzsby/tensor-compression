from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    require_args,
    resolve_path_string,
    set_default,
    value_to_csv,
)


STRUCTURED_QUERY_FEATURE_DIM = 32
LABELS = [f"B{index:02d}" for index in range(10)] + ["A", "B"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def _normalize_coordinate(value: Any, size: int) -> float:
    if size <= 1:
        return 0.0
    clipped = max(0.0, min(float(value), float(size - 1)))
    return (clipped / float(size - 1)) * 2.0 - 1.0


def _normalize_length(value: Any, size: int) -> float:
    if size <= 0:
        return 0.0
    clipped = max(0.0, min(float(value), float(size)))
    return clipped / float(size)


def structured_query_features_for_record(record: Mapping[str, Any]) -> list[float]:
    """Structured features copied from the LLM adapter path, without using oracle values."""
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    grid_shape = metadata.get("grid_shape") if isinstance(metadata, Mapping) else None
    has_grid_shape = isinstance(grid_shape, Sequence) and not isinstance(grid_shape, str)
    height = int(grid_shape[0]) if has_grid_shape and len(grid_shape) >= 1 else 512
    width = int(grid_shape[1]) if has_grid_shape and len(grid_shape) >= 2 else 512
    task_type = str(record.get("task_type", ""))
    query = str(record.get("query") or record.get("question") or "")
    choices = record.get("choices")
    choice_count = len(choices) if isinstance(choices, Sequence) and not isinstance(choices, str) else 0

    features = [0.0] * STRUCTURED_QUERY_FEATURE_DIM
    task_order = ["point_bin", "point_compare", "patch_compare", "max_speed_quadrant", "global_stat_bin"]
    if task_type in task_order:
        features[task_order.index(task_type)] = 1.0
    features[5] = _normalize_length(choice_count, 16)
    features[6] = 1.0 if "Vx" in query else 0.0
    features[7] = 1.0 if "Vy" in query else 0.0

    point = re.search(r"row=(\d+)\s+col=(\d+)", query)
    if point:
        row = int(point.group(1))
        col = int(point.group(2))
        features[8] = _normalize_coordinate(row, height)
        features[9] = _normalize_coordinate(col, width)
        features[10] = _normalize_length(row // max(1, height // 16), 16)
        features[11] = _normalize_length(col // max(1, width // 16), 16)

    point_pair = re.search(r"A=\((\d+),(\d+)\)\s+B=\((\d+),(\d+)\)", query)
    if point_pair:
        row_a, col_a, row_b, col_b = [int(group) for group in point_pair.groups()]
        features[12] = _normalize_coordinate(row_a, height)
        features[13] = _normalize_coordinate(col_a, width)
        features[14] = _normalize_coordinate(row_b, height)
        features[15] = _normalize_coordinate(col_b, width)
        features[16] = _normalize_coordinate(row_b - row_a + (height - 1) / 2.0, height)
        features[17] = _normalize_coordinate(col_b - col_a + (width - 1) / 2.0, width)

    patch_pair = re.search(
        r"A=\[(\d+):(\d+),(\d+):(\d+)\]\s+B=\[(\d+):(\d+),(\d+):(\d+)\]",
        query,
    )
    if patch_pair:
        row0_a, row1_a, col0_a, col1_a, row0_b, row1_b, col0_b, col1_b = [
            int(group) for group in patch_pair.groups()
        ]
        center_row_a = (row0_a + row1_a - 1) / 2.0
        center_col_a = (col0_a + col1_a - 1) / 2.0
        center_row_b = (row0_b + row1_b - 1) / 2.0
        center_col_b = (col0_b + col1_b - 1) / 2.0
        features[18] = _normalize_coordinate(center_row_a, height)
        features[19] = _normalize_coordinate(center_col_a, width)
        features[20] = _normalize_coordinate(center_row_b, height)
        features[21] = _normalize_coordinate(center_col_b, width)
        features[22] = _normalize_length(row1_a - row0_a, height)
        features[23] = _normalize_length(col1_a - col0_a, width)
        features[24] = _normalize_length(row1_b - row0_b, height)
        features[25] = _normalize_length(col1_b - col0_b, width)
    return features


def resolve_device(value: str | None) -> torch.device:
    if value is None or value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def dump_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def qa_jsonl_path(qa_dir: str | Path, split: str) -> Path:
    root = Path(qa_dir)
    if root.is_file():
        return root
    candidates = [
        root / f"{split}.jsonl",
        root / f"{split}.json",
        root / split,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / f"{split}.jsonl"


def label_index(label: Any) -> int:
    text = str(label)
    if text not in LABEL_TO_INDEX:
        raise ValueError(f"Unsupported answer label {text!r}; expected one of {LABELS}.")
    return int(LABEL_TO_INDEX[text])


def choice_indices_for_record(record: Mapping[str, Any]) -> list[int]:
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        choices = [record["answer"]]
    labels = [str(choice) for choice in choices]
    answer = str(record["answer"])
    if answer not in labels:
        labels = [answer] + labels
    return [label_index(label) for label in labels]


class TensorReadoutProbeDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        latent_dir: str | Path,
        max_records: int | None = None,
        prefer_record_latent_ref: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.latent_dir = Path(latent_dir)
        self.prefer_record_latent_ref = bool(prefer_record_latent_ref)
        self.records = self._load_records(self.jsonl_path)
        if max_records is not None:
            self.records = self.records[: max(0, int(max_records))]
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.jsonl_path}.")
        self._random_different_indices = self._build_random_different_indices(int(shuffle_seed))

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}.")
                records.append(payload)
        return records

    def _build_random_different_indices(self, seed: int) -> list[int]:
        total = len(self.records)
        unique_states = {str(record.get("state_ref", "")) for record in self.records}
        if len(unique_states) < 2:
            return list(range(total))
        rng = random.Random(seed)
        indices: list[int] = []
        for index, record in enumerate(self.records):
            state_ref = str(record.get("state_ref", ""))
            candidate = rng.randrange(total)
            attempts = 0
            while str(self.records[candidate].get("state_ref", "")) == state_ref:
                candidate = rng.randrange(total)
                attempts += 1
                if attempts > total * 4:
                    candidate = index
                    break
            indices.append(int(candidate))
        return indices

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        return {
            "index": int(index),
            "record": record,
            "latent_map": self.load_latent_for_record(record),
            "query_features": torch.tensor(structured_query_features_for_record(record), dtype=torch.float32),
            "target": label_index(record["answer"]),
            "choice_indices": choice_indices_for_record(record),
        }

    def latent_path_for_record(self, record: Mapping[str, Any]) -> Path:
        state_ref = str(record.get("state_ref") or "")
        if not state_ref:
            sample_index = int(record["sample_index"])
            time_index = int(record["time_index"])
            state_ref = f"sample{sample_index:06d}_t{time_index:04d}"
        latent_from_dir = self.latent_dir / f"{state_ref}.pt"
        record_ref = record.get("latent_ref")
        if self.prefer_record_latent_ref and record_ref:
            return Path(str(record_ref))
        if latent_from_dir.exists():
            return latent_from_dir
        if record_ref:
            return Path(str(record_ref))
        return latent_from_dir

    def load_latent_for_record(self, record: Mapping[str, Any]) -> torch.Tensor:
        path = self.latent_path_for_record(record)
        if not path.exists():
            raise FileNotFoundError(f"Latent cache file not found: {path}")
        payload = torch.load(path, map_location="cpu")
        latent = payload.get("latent_map") if isinstance(payload, Mapping) else payload
        if not isinstance(latent, torch.Tensor):
            raise ValueError(f"Latent cache file does not contain a tensor latent_map: {path}")
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim != 3:
            raise ValueError(f"Expected latent_map [C,H,W], got {tuple(latent.shape)} from {path}")
        return latent.to(dtype=torch.float32)

    def load_shuffled_latent(self, index: int) -> torch.Tensor:
        other_index = self._random_different_indices[int(index)]
        return self.load_latent_for_record(self.records[other_index])


def collate_probe(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "indices": [int(item["index"]) for item in items],
        "records": [item["record"] for item in items],
        "latent_map": torch.stack([item["latent_map"] for item in items], dim=0),
        "query_features": torch.stack([item["query_features"] for item in items], dim=0),
        "target": torch.tensor([int(item["target"]) for item in items], dtype=torch.long),
        "choice_indices": [list(item["choice_indices"]) for item in items],
    }


class TensorLocalDirectProbe(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        query_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
        feature_mode: str,
    ) -> None:
        super().__init__()
        if feature_mode not in {"global", "local", "local_global"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        self.latent_channels = int(latent_channels)
        self.query_dim = int(query_dim)
        self.feature_mode = str(feature_mode)
        latent_feature_blocks = 0
        if self.feature_mode in {"global", "local_global"}:
            latent_feature_blocks += 2
        if self.feature_mode in {"local", "local_global"}:
            latent_feature_blocks += 9
        input_dim = int(latent_feature_blocks * self.latent_channels + self.query_dim)

        modules: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for _ in range(max(1, int(hidden_layers))):
            modules.extend(
                [
                    nn.Linear(current_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = int(hidden_dim)
        modules.append(nn.Linear(current_dim, len(LABELS)))
        self.classifier = nn.Sequential(*modules)

    @staticmethod
    def _sample_points(latent_map: torch.Tensor, coords_row_col: torch.Tensor) -> torch.Tensor:
        coords = coords_row_col.clamp(-1.0, 1.0)
        grid = torch.stack([coords[..., 1], coords[..., 0]], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            latent_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def _patch_pool(
        self,
        latent_map: torch.Tensor,
        center_row_col: torch.Tensor,
        half_extent_row_col: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            device=latent_map.device,
            dtype=latent_map.dtype,
        )
        coords = center_row_col.unsqueeze(1) + offsets.unsqueeze(0) * half_extent_row_col.unsqueeze(1)
        return self._sample_points(latent_map, coords).mean(dim=1)

    def _local_features(self, latent_map: torch.Tensor, query_features: torch.Tensor) -> list[torch.Tensor]:
        q = query_features.to(device=latent_map.device, dtype=latent_map.dtype)
        point = self._sample_points(latent_map, q[:, 8:10].unsqueeze(1)).squeeze(1)
        point_a = self._sample_points(latent_map, q[:, 12:14].unsqueeze(1)).squeeze(1)
        point_b = self._sample_points(latent_map, q[:, 14:16].unsqueeze(1)).squeeze(1)

        patch_a_center = q[:, 18:20]
        patch_b_center = q[:, 20:22]
        # L / image_size is approximately the patch half-width in [-1, 1] grid_sample coordinates.
        patch_a_half = q[:, 22:24]
        patch_b_half = q[:, 24:26]
        patch_a = self._patch_pool(latent_map, patch_a_center, patch_a_half)
        patch_b = self._patch_pool(latent_map, patch_b_center, patch_b_half)

        return [
            point,
            point_a,
            point_b,
            point_a - point_b,
            torch.abs(point_a - point_b),
            patch_a,
            patch_b,
            patch_a - patch_b,
            torch.abs(patch_a - patch_b),
        ]

    def forward(self, latent_map: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        if latent_map.ndim != 4:
            raise ValueError(f"Expected latent_map [B,C,H,W], got {tuple(latent_map.shape)}.")
        latent_map = latent_map.to(dtype=torch.float32)
        q = query_features.to(device=latent_map.device, dtype=torch.float32)

        parts: list[torch.Tensor] = []
        if self.feature_mode in {"global", "local_global"}:
            parts.extend(
                [
                    latent_map.mean(dim=(2, 3)),
                    latent_map.std(dim=(2, 3), unbiased=False),
                ]
            )
        if self.feature_mode in {"local", "local_global"}:
            parts.extend(self._local_features(latent_map, q))
        parts.append(q)
        features = torch.cat(parts, dim=-1)
        return self.classifier(features)


def choice_ce_loss(
    logits: torch.Tensor,
    choice_indices: Sequence[Sequence[int]],
    target_indices: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row, choices in enumerate(choice_indices):
        choices_tensor = torch.tensor(list(choices), dtype=torch.long, device=logits.device)
        choice_logits = logits[row].index_select(0, choices_tensor).unsqueeze(0)
        target = int(target_indices[row].item())
        try:
            target_position = list(choices).index(target)
        except ValueError as exc:
            raise ValueError(f"Target label index {target} is missing from choices {list(choices)}.") from exc
        target_tensor = torch.tensor([target_position], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(choice_logits, target_tensor))
    return torch.stack(losses).mean()


def choice_predictions(logits: torch.Tensor, choice_indices: Sequence[Sequence[int]]) -> list[int]:
    predictions: list[int] = []
    for row, choices in enumerate(choice_indices):
        choices_list = list(choices)
        choices_tensor = torch.tensor(choices_list, dtype=torch.long, device=logits.device)
        best_position = int(logits[row].index_select(0, choices_tensor).argmax().item())
        predictions.append(int(choices_list[best_position]))
    return predictions


def baseline_latents(mode: str, batch: Mapping[str, Any], dataset: TensorReadoutProbeDataset) -> torch.Tensor:
    latents = batch["latent_map"]
    if mode == "correct":
        return latents
    if mode == "zero_latent":
        return torch.zeros_like(latents)
    if mode == "random":
        return torch.randn_like(latents)
    if mode == "shuffled":
        return torch.stack([dataset.load_shuffled_latent(index) for index in batch["indices"]], dim=0)
    raise ValueError(f"Unsupported eval baseline: {mode}")


def update_task_metrics(
    metrics: dict[str, Any],
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[int],
    targets: Sequence[int],
) -> None:
    for record, prediction, target in zip(records, predictions, targets):
        task_type = str(record.get("task_type", "unknown"))
        hit = int(int(prediction) == int(target))
        metrics["correct"] += hit
        metrics["total"] += 1
        task_bucket = metrics["by_task"].setdefault(task_type, {"correct": 0, "total": 0})
        task_bucket["correct"] += hit
        task_bucket["total"] += 1


def finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    output = {
        "accuracy": float(metrics["correct"] / max(1, metrics["total"])),
        "correct": int(metrics["correct"]),
        "total": int(metrics["total"]),
        "by_task": {},
    }
    for task_type, task_metrics in sorted(metrics["by_task"].items()):
        output["by_task"][task_type] = {
            "accuracy": float(task_metrics["correct"] / max(1, task_metrics["total"])),
            "correct": int(task_metrics["correct"]),
            "total": int(task_metrics["total"]),
        }
    return output


def train_one_epoch(
    model: TensorLocalDirectProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
    epoch: int,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_records = 0
    correct = 0
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        latent_map = batch["latent_map"].to(device)
        query_features = batch["query_features"].to(device)
        targets = batch["target"].to(device)
        logits = model(latent_map, query_features)
        loss = choice_ce_loss(logits, batch["choice_indices"], targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
        optimizer.step()

        predictions = choice_predictions(logits.detach(), batch["choice_indices"])
        target_list = [int(value) for value in targets.detach().cpu().tolist()]
        batch_correct = sum(int(prediction == target) for prediction, target in zip(predictions, target_list))
        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_records += batch_size
        correct += batch_correct
        if log_interval > 0 and step % int(log_interval) == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(1, total_records):.4f}",
                acc=f"{correct / max(1, total_records):.4f}",
            )
    return {
        "loss": float(total_loss / max(1, total_records)),
        "accuracy": float(correct / max(1, total_records)),
    }


@torch.no_grad()
def evaluate(
    model: TensorLocalDirectProbe,
    dataset: TensorReadoutProbeDataset,
    loader: DataLoader,
    device: torch.device,
    modes: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    outputs: dict[str, Any] = {}
    for mode in modes:
        loss_sum = 0.0
        total_records = 0
        metrics = {"correct": 0, "total": 0, "by_task": {}}
        for batch in tqdm(loader, desc=f"eval {mode}", leave=False):
            latent_map = baseline_latents(mode, batch, dataset).to(device)
            query_features = batch["query_features"].to(device)
            targets = batch["target"].to(device)
            logits = model(latent_map, query_features)
            loss = choice_ce_loss(logits, batch["choice_indices"], targets)
            predictions = choice_predictions(logits, batch["choice_indices"])
            batch_size = int(targets.numel())
            loss_sum += float(loss.item()) * batch_size
            total_records += batch_size
            target_list = [int(value) for value in targets.detach().cpu().tolist()]
            update_task_metrics(metrics, batch["records"], predictions, target_list)
        finalized = finalize_metrics(metrics)
        finalized["loss"] = float(loss_sum / max(1, total_records))
        outputs[str(mode)] = finalized
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a direct probe on cached tensor latents. This bypasses the frozen LLM and tests "
            "whether the AE latent itself contains enough information for tensor readout QA."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional tensor-LLM pipeline YAML config.")
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--source-split", type=str, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument(
        "--overfit-records",
        type=int,
        default=None,
        help="If set, use --source-split for train/val/test and cap all three to this many records.",
    )
    parser.add_argument("--prefer-record-latent-ref", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--feature-mode",
        type=str,
        choices=("global", "local", "local_global"),
        default=None,
        help="global: pooled latent only; local: query-addressed latent samples only; local_global: both.",
    )
    parser.add_argument(
        "--eval-baselines",
        type=str,
        default=None,
        help="Comma-separated: correct,zero_latent,shuffled,random.",
    )
    parser.add_argument("--log-interval", type=int, default=None)
    return parser.parse_args()


def apply_config_defaults(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    set_default(args, "qa_dir", first_nested(config, ["data.qa_dir"]))
    set_default(args, "latent_dir", first_nested(config, ["data.latent_dir", "latent_export.output_dir"]))

    output_root = first_nested(config, ["direct_probe.output_root", "llm_training.output_root"])
    if output_root is None:
        storage_output_root = first_nested(config, ["storage.output_root"])
        output_root = str(Path(str(storage_output_root)) / "runs") if storage_output_root is not None else "./outputs/runs"
    set_default(args, "output_root", output_root)

    set_default(args, "run_name", first_nested(config, ["direct_probe.run_name"]), "tensor_direct_probe")
    set_default(args, "train_split", first_nested(config, ["direct_probe.train_split", "llm_training.train_split"]), "train")
    set_default(args, "val_split", first_nested(config, ["direct_probe.val_split", "llm_training.val_split"]), "val")
    set_default(args, "test_split", first_nested(config, ["direct_probe.test_split", "llm_training.test_split"]), "test")
    set_default(args, "source_split", first_nested(config, ["direct_probe.source_split"]), "train")
    set_default(
        args,
        "max_train_records",
        first_nested(config, ["direct_probe.max_train_records", "llm_training.max_train_records"]),
    )
    set_default(
        args,
        "max_val_records",
        first_nested(config, ["direct_probe.max_val_records", "llm_training.max_val_records"]),
    )
    set_default(
        args,
        "max_test_records",
        first_nested(config, ["direct_probe.max_test_records", "llm_training.max_test_records"]),
    )
    set_default(
        args,
        "prefer_record_latent_ref",
        first_nested(config, ["direct_probe.prefer_record_latent_ref", "llm_training.prefer_record_latent_ref"]),
        False,
    )
    set_default(args, "device", first_nested(config, ["direct_probe.device", "runtime.device"]), "auto")
    set_default(args, "seed", first_nested(config, ["direct_probe.seed", "runtime.seed"]), 42)
    set_default(args, "shuffle_seed", first_nested(config, ["direct_probe.shuffle_seed", "llm_training.shuffle_seed"]), 42)
    set_default(args, "epochs", first_nested(config, ["direct_probe.epochs"]), 50)
    set_default(args, "batch_size", first_nested(config, ["direct_probe.batch_size"]), 32)
    set_default(args, "eval_batch_size", first_nested(config, ["direct_probe.eval_batch_size"]), args.batch_size)
    set_default(args, "num_workers", first_nested(config, ["direct_probe.num_workers"]), 0)
    set_default(args, "lr", first_nested(config, ["direct_probe.lr"]), 1.0e-3)
    set_default(args, "weight_decay", first_nested(config, ["direct_probe.weight_decay"]), 1.0e-4)
    set_default(args, "grad_clip_norm", first_nested(config, ["direct_probe.grad_clip_norm"]), 1.0)
    set_default(args, "hidden_dim", first_nested(config, ["direct_probe.hidden_dim"]), 512)
    set_default(args, "hidden_layers", first_nested(config, ["direct_probe.hidden_layers"]), 2)
    set_default(args, "dropout", first_nested(config, ["direct_probe.dropout"]), 0.0)
    set_default(args, "feature_mode", first_nested(config, ["direct_probe.feature_mode"]), "local_global")
    set_default(
        args,
        "eval_baselines",
        value_to_csv(first_nested(config, ["direct_probe.eval_baselines"])),
        "correct,zero_latent,shuffled",
    )
    set_default(args, "log_interval", first_nested(config, ["direct_probe.log_interval"]), 20)

    if args.overfit_records is not None:
        source_split = str(args.source_split or "train")
        args.train_split = source_split
        args.val_split = source_split
        args.test_split = source_split
        args.max_train_records = int(args.overfit_records)
        args.max_val_records = int(args.overfit_records)
        args.max_test_records = int(args.overfit_records)

    for attr in ("qa_dir", "latent_dir", "output_root"):
        setattr(args, attr, resolve_path_string(getattr(args, attr), PROJECT_ROOT))
    require_args(args, ["qa_dir", "latent_dir", "output_root"])
    return args


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_probe,
    )


def save_probe_checkpoint(
    path: Path,
    model: TensorLocalDirectProbe,
    args: argparse.Namespace,
    latent_shape: Sequence[int],
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "labels": LABELS,
        "latent_shape_chw": [int(dim) for dim in latent_shape],
        "epoch": int(epoch),
        "metrics": metrics,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = load_yaml_mapping(args.config)
    args = apply_config_defaults(args, config)
    seed_everything(int(args.seed))
    device = resolve_device(str(args.device))

    eval_modes = [mode.strip() for mode in str(args.eval_baselines).split(",") if mode.strip()]
    if "correct" not in eval_modes:
        eval_modes = ["correct"] + eval_modes

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    dump_json(run_dir / "args.json", vars(args))

    train_dataset = TensorReadoutProbeDataset(
        qa_jsonl_path(args.qa_dir, str(args.train_split)),
        latent_dir=args.latent_dir,
        max_records=args.max_train_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )
    val_dataset = TensorReadoutProbeDataset(
        qa_jsonl_path(args.qa_dir, str(args.val_split)),
        latent_dir=args.latent_dir,
        max_records=args.max_val_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )
    test_dataset = TensorReadoutProbeDataset(
        qa_jsonl_path(args.qa_dir, str(args.test_split)),
        latent_dir=args.latent_dir,
        max_records=args.max_test_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )

    first_latent = train_dataset[0]["latent_map"]
    latent_shape = tuple(int(dim) for dim in first_latent.shape)
    if len(latent_shape) != 3:
        raise ValueError(f"Expected first latent shape [C,H,W], got {latent_shape}.")

    model = TensorLocalDirectProbe(
        latent_channels=int(latent_shape[0]),
        query_dim=STRUCTURED_QUERY_FEATURE_DIM,
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        dropout=float(args.dropout),
        feature_mode=str(args.feature_mode),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_loader = make_loader(train_dataset, int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = make_loader(val_dataset, int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers))
    test_loader = make_loader(test_dataset, int(args.eval_batch_size), shuffle=False, num_workers=int(args.num_workers))

    run_summary = {
        "device": str(device),
        "latent_shape_chw": list(latent_shape),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "test_records": len(test_dataset),
        "labels": LABELS,
        "feature_mode": str(args.feature_mode),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "trainable_probe_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    dump_json(run_dir / "run_summary.json", run_summary)

    metrics_history: dict[str, Any] = {}
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=float(args.grad_clip_norm),
            epoch=epoch,
            log_interval=int(args.log_interval),
        )
        val_metrics = evaluate(model, val_dataset, val_loader, device, eval_modes)
        epoch_metrics = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics_history[f"epoch_{epoch:04d}"] = epoch_metrics
        dump_json(run_dir / "metrics_latest.json", metrics_history)
        save_probe_checkpoint(
            run_dir / "probe_last.pt",
            model=model,
            args=args,
            latent_shape=latent_shape,
            epoch=epoch,
            metrics=epoch_metrics,
        )
        correct_accuracy = float(val_metrics["correct"]["accuracy"])
        if correct_accuracy > best_accuracy:
            best_accuracy = correct_accuracy
            best_epoch = epoch
            save_probe_checkpoint(
                run_dir / "probe_best.pt",
                model=model,
                args=args,
                latent_shape=latent_shape,
                epoch=epoch,
                metrics=epoch_metrics,
            )
        print(
            f"epoch={epoch:04d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_correct={val_metrics['correct']['accuracy']:.4f}"
        )

    best_checkpoint = torch.load(run_dir / "probe_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_dataset, test_loader, device, eval_modes)
    dump_json(run_dir / "test_metrics.json", test_metrics)
    metrics_history["best"] = {"epoch": int(best_epoch), "val_correct_accuracy": float(best_accuracy)}
    metrics_history["test"] = test_metrics
    dump_json(run_dir / "metrics_latest.json", metrics_history)
    print(f"Run directory: {run_dir}")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
