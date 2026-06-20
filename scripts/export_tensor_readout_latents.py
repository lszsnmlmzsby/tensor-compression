from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.config import load_config  # noqa: E402
from tensor_compression.data.normalization import normalize_tensor  # noqa: E402
from tensor_compression.downstream.pdebench import (  # noqa: E402
    read_pdebench_sample,
    resolve_checkpoint_field_keys,
    resolve_device,
    resize_chw_batch,
    validate_checkpoint_field_keys_against_model,
)
from tensor_compression.models import build_model  # noqa: E402
from tensor_compression.utils import dump_json  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    require_args,
    resolve_path_string,
    set_default,
    value_to_csv,
)


@dataclass(frozen=True)
class StateRequest:
    state_ref: str
    sample_index: int
    time_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one latent-map cache file per tensor state used by tensor-readout QA. "
            "The compressor is frozen; this script only runs its encoder."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional tensor-LLM pipeline YAML config.")
    parser.add_argument("--qa-dir", type=str, default=None, help="Directory with train/val/test JSONL files.")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated split names to scan from --qa-dir.",
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        default=None,
        help="PDEBench HDF5 path. If omitted, metadata.json in --qa-dir is used.",
    )
    parser.add_argument("--compressor-checkpoint", type=str, default=None, help="AE checkpoint path.")
    parser.add_argument(
        "--compressor-config",
        type=str,
        default=None,
        help="Optional AE config path when the checkpoint does not contain config.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Optional comma-separated field keys. Defaults to checkpoint/config field order.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for <state_ref>.pt latent files.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--spatial-stride", type=int, default=None)
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default=None,
        choices=("float32", "float16", "bfloat16"),
        help="Dtype used when saving latent cache files.",
    )
    parser.add_argument(
        "--allow-field-mismatch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow QA metadata fields to differ from compressor fields. "
            "Use only if every question is known to query fields encoded by the compressor."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Re-export latent files that already exist.",
    )
    return apply_config_defaults(parser.parse_args())


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_mapping(args.config)

    path_defaults = {
        "qa_dir": first_nested(config, ["data.qa_dir"]),
        "hdf5_path": first_nested(config, ["data.hdf5_path"]),
        "compressor_checkpoint": first_nested(config, ["compressor.checkpoint"]),
        "compressor_config": first_nested(config, ["compressor.config"]),
        "output_dir": first_nested(config, ["latent_export.output_dir", "data.latent_dir"]),
    }
    for attr, value in path_defaults.items():
        if getattr(args, attr, None) is None and value is not None:
            setattr(args, attr, resolve_path_string(value, PROJECT_ROOT))

    set_default(args, "splits", value_to_csv(first_nested(config, ["latent_export.splits"])), "train,val,test")
    set_default(args, "fields", value_to_csv(first_nested(config, ["data.fields"])), None)
    set_default(args, "batch_size", first_nested(config, ["latent_export.batch_size"]), 4)
    set_default(args, "device", first_nested(config, ["latent_export.device", "runtime.device"]), "auto")
    set_default(
        args,
        "spatial_stride",
        first_nested(config, ["latent_export.spatial_stride", "qa_generation.spatial_stride"]),
        1,
    )
    set_default(args, "storage_dtype", first_nested(config, ["latent_export.storage_dtype"]), "float16")
    set_default(args, "allow_field_mismatch", first_nested(config, ["latent_export.allow_field_mismatch"]), False)
    set_default(args, "overwrite", first_nested(config, ["latent_export.overwrite"]), False)
    require_args(args, ["qa_dir", "compressor_checkpoint", "output_dir"])
    return args


def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            records.append(payload)
    return records


def qa_paths(qa_dir: Path, splits: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        path = qa_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"QA split file not found: {path}")
        paths.append(path)
    return paths


def infer_hdf5_path(qa_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    metadata_path = qa_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("Pass --hdf5-path or provide metadata.json in --qa-dir.")
    metadata = load_json(metadata_path)
    source = metadata.get("source")
    if not isinstance(source, Mapping) or not source.get("hdf5_path"):
        raise ValueError(f"metadata.json does not contain source.hdf5_path: {metadata_path}")
    return Path(str(source["hdf5_path"])).expanduser()


def load_checkpoint_and_config(
    checkpoint_path: str | Path,
    config_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if config_path is not None:
        config = load_config(config_path, base_root=PROJECT_ROOT)
    else:
        raw_config = checkpoint.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(
                "Compressor checkpoint does not contain config. "
                "Pass --compressor-config explicitly."
            )
        config = dict(raw_config)
    return dict(checkpoint), config


def resolve_field_keys(args: argparse.Namespace, config: Mapping[str, Any]) -> list[str]:
    cli_fields = parse_csv(args.fields)
    checkpoint_fields = resolve_checkpoint_field_keys(config)
    if cli_fields and checkpoint_fields and cli_fields != checkpoint_fields:
        raise ValueError(
            "Provided --fields differs from the compressor config field order. "
            f"CLI={cli_fields}, checkpoint={checkpoint_fields}."
        )
    fields = cli_fields or checkpoint_fields
    if not fields:
        raise ValueError("Could not resolve compressor field keys. Pass --fields explicitly.")
    validate_checkpoint_field_keys_against_model(config, fields)
    return [str(field) for field in fields]


def collect_state_requests(records: Sequence[dict[str, Any]]) -> list[StateRequest]:
    requests: dict[str, StateRequest] = {}
    for record in records:
        state_ref = str(record.get("state_ref") or "")
        if not state_ref:
            sample_index = int(record["sample_index"])
            time_index = int(record["time_index"])
            state_ref = f"sample{sample_index:06d}_t{time_index:04d}"
        request = StateRequest(
            state_ref=state_ref,
            sample_index=int(record["sample_index"]),
            time_index=int(record["time_index"]),
        )
        previous = requests.get(request.state_ref)
        if previous is not None and previous != request:
            raise ValueError(
                f"Conflicting sample/time metadata for state_ref={request.state_ref}: "
                f"{previous} vs {request}"
            )
        requests[request.state_ref] = request
    return sorted(requests.values(), key=lambda item: (item.sample_index, item.time_index, item.state_ref))


def validate_record_fields(
    records: Sequence[dict[str, Any]],
    compressor_fields: Sequence[str],
    allow_mismatch: bool,
) -> None:
    seen: set[tuple[str, ...]] = set()
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        fields = metadata.get("fields")
        if isinstance(fields, Sequence) and not isinstance(fields, str):
            seen.add(tuple(str(field) for field in fields))
    expected = tuple(str(field) for field in compressor_fields)
    mismatches = sorted(fields for fields in seen if fields != expected)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "QA records were generated with fields that differ from the compressor fields. "
            f"compressor_fields={list(expected)}, qa_fields={mismatches}. "
            "Regenerate QA with matching --fields, or pass --allow-field-mismatch only for a controlled ablation."
        )


def storage_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def normalize_frames(frames: torch.Tensor, normalization_cfg: Mapping[str, Any] | None) -> torch.Tensor:
    normalized = []
    for frame in frames:
        normalized_frame, _state = normalize_tensor(frame.cpu(), dict(normalization_cfg or {}))
        normalized.append(normalized_frame)
    return torch.stack(normalized, dim=0)


def save_latent(
    path: Path,
    latent_map: torch.Tensor,
    request: StateRequest,
    field_keys: Sequence[str],
    checkpoint_path: str | Path,
    config_path: str | Path | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latent_map": latent_map.detach().cpu(),
        "sample_index": int(request.sample_index),
        "time_index": int(request.time_index),
        "state_ref": request.state_ref,
        "field_keys": list(field_keys),
        "compressor_checkpoint": str(checkpoint_path),
        "compressor_config": str(config_path) if config_path is not None else None,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    qa_dir = Path(args.qa_dir)
    splits = parse_csv(args.splits)
    if not splits:
        raise ValueError("--splits must contain at least one split name.")

    paths = qa_paths(qa_dir, splits)
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    if not records:
        raise RuntimeError(f"No QA records found in {qa_dir}.")

    checkpoint, config = load_checkpoint_and_config(args.compressor_checkpoint, args.compressor_config)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {args.compressor_checkpoint}")

    field_keys = resolve_field_keys(args, config)
    validate_record_fields(records, field_keys, bool(args.allow_field_mismatch))

    model = build_model(config)
    model.load_state_dict(state_dict)
    device = resolve_device(args.device)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hdf5_path = infer_hdf5_path(qa_dir, args.hdf5_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = collect_state_requests(records)
    requests_by_sample: dict[int, list[StateRequest]] = defaultdict(list)
    for request in requests:
        requests_by_sample[request.sample_index].append(request)

    input_size = tuple(int(dim) for dim in config["model"]["input_size"])
    normalization_cfg = dict(config.get("data", {}).get("dataset", {}).get("normalization", {}))
    batch_size = max(1, int(args.batch_size))
    target_dtype = storage_dtype(str(args.storage_dtype))
    exported = 0
    skipped = 0
    latent_shape: list[int] | None = None

    with torch.no_grad():
        for sample_index, sample_requests in tqdm(
            sorted(requests_by_sample.items()),
            desc="Export latents",
            unit="sample",
        ):
            data, _grid, _t_coordinates = read_pdebench_sample(
                hdf5_path=hdf5_path,
                field_keys=field_keys,
                sample_index=sample_index,
                spatial_stride=int(args.spatial_stride),
            )
            for start in range(0, len(sample_requests), batch_size):
                batch_requests = sample_requests[start : start + batch_size]
                pending_requests: list[StateRequest] = []
                frames: list[torch.Tensor] = []
                for request in batch_requests:
                    output_path = output_dir / f"{request.state_ref}.pt"
                    if output_path.exists() and not args.overwrite:
                        if latent_shape is None:
                            existing = torch.load(output_path, map_location="cpu")
                            existing_latent = existing.get("latent_map") if isinstance(existing, Mapping) else existing
                            if isinstance(existing_latent, torch.Tensor):
                                latent_shape = [int(dim) for dim in existing_latent.shape]
                        skipped += 1
                        continue
                    if request.time_index < 0 or request.time_index >= data.shape[2]:
                        raise IndexError(
                            f"time_index={request.time_index} is outside sample data time axis "
                            f"with length {data.shape[2]} for sample {sample_index}."
                        )
                    frame = data[:, :, request.time_index, :].permute(2, 0, 1).contiguous()
                    frames.append(frame)
                    pending_requests.append(request)
                if not pending_requests:
                    continue

                frame_batch = torch.stack(frames, dim=0).to(dtype=torch.float32)
                frame_batch = resize_chw_batch(frame_batch, input_size)
                normalized = normalize_frames(frame_batch, normalization_cfg).to(device=device, dtype=torch.float32)
                outputs = model.encode(normalized)
                latent_maps = outputs["latent_map"].detach().cpu()
                if latent_shape is None:
                    latent_shape = [int(dim) for dim in latent_maps.shape[1:]]
                for latent_map, request in zip(latent_maps, pending_requests):
                    save_latent(
                        path=output_dir / f"{request.state_ref}.pt",
                        latent_map=latent_map.to(dtype=target_dtype),
                        request=request,
                        field_keys=field_keys,
                        checkpoint_path=args.compressor_checkpoint,
                        config_path=args.compressor_config,
                    )
                    exported += 1

    manifest = {
        "qa_dir": str(qa_dir),
        "splits": splits,
        "hdf5_path": str(hdf5_path),
        "compressor_checkpoint": str(args.compressor_checkpoint),
        "compressor_config": str(args.compressor_config) if args.compressor_config else None,
        "field_keys": field_keys,
        "state_count": len(requests),
        "exported": exported,
        "skipped_existing": skipped,
        "latent_shape_chw": latent_shape,
        "storage_dtype": str(args.storage_dtype),
        "spatial_stride": int(args.spatial_stride),
    }
    dump_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
