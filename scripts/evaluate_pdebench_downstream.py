from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import (
    CheckpointReconstructor,
    ProgressEvent,
    build_operator,
    emit_progress,
    evaluate_records,
    generate_pdebench_records,
    inspect_pdebench_fields,
    parse_fields,
    parse_time_slice,
    prepare_reconstructed_hdf5_output,
    resolve_checkpoint_field_keys,
    resolve_field_keys_for_evaluation,
    summarize_metrics,
    validate_checkpoint_field_keys_against_model,
    write_reconstructed_record_to_hdf5,
)
from tensor_compression.utils import dump_json
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PDEBench operator outputs on original vs compressor-reconstructed data."
        )
    )
    parser.add_argument("--hdf5-path", type=str, required=True, help="PDEBench HDF5 file.")
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help=(
            "Comma-separated HDF5 field keys. If a compressor checkpoint is provided, "
            "the training field order is used automatically by default, and any explicit "
            "--fields must exactly match that order."
        ),
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0",
        help="Comma-separated sample indices to evaluate, e.g. 0,1,2. Use 'all' for all samples.",
    )
    parser.add_argument("--time-start", type=int, default=None)
    parser.add_argument("--time-stop", type=int, default=None)
    parser.add_argument("--time-step", type=int, default=None)
    parser.add_argument("--spatial-stride", type=int, default=1)
    parser.add_argument(
        "--compressor-checkpoint",
        type=str,
        default=None,
        help="Path to trained AE checkpoint. If omitted, only identity reconstruction is evaluated.",
    )
    parser.add_argument(
        "--compressor-config",
        type=str,
        default=None,
        help="Optional compressor config if the checkpoint lacks an embedded config.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--forward-operator-type",
        choices=("none", "callable", "pdebench-fno", "pdebench-unet"),
        default="none",
    )
    parser.add_argument(
        "--forward-operator-spec",
        type=str,
        default=None,
        help="For callable: module.py:callable or import.path:callable.",
    )
    parser.add_argument("--forward-checkpoint", type=str, default=None)

    parser.add_argument(
        "--inverse-operator-type",
        choices=("none", "callable"),
        default="none",
        help=(
            "Inverse evaluation currently expects a callable wrapper. "
            "Use module.py:callable that consumes the standard payload."
        ),
    )
    parser.add_argument(
        "--inverse-operator-spec",
        type=str,
        default=None,
        help="For callable inverse operator: module.py:callable or import.path:callable.",
    )

    parser.add_argument(
        "--pdebench-root",
        type=str,
        default=str(PROJECT_ROOT / "PDEBench_code" / "PDEBench-main"),
        help="PDEBench repository root, required for pdebench-fno/unet operators.",
    )
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--initial-step", type=int, default=10)
    parser.add_argument("--t-train", type=int, default=None)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--init-features", type=int, default=32)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to outputs/pdebench_downstream/<timestamp>.json.",
    )
    parser.add_argument(
        "--reconstructed-hdf5-output",
        type=str,
        default=None,
        help=(
            "Optional output HDF5 path. The source file is copied, selected fields are "
            "replaced with compressor reconstructions, and all other datasets/metadata stay unchanged."
        ),
    )
    parser.add_argument(
        "--overwrite-reconstructed-hdf5",
        action="store_true",
        help="Allow overwriting --reconstructed-hdf5-output if it already exists.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display and stage updates during evaluation.",
    )
    return parser.parse_args()


def parse_sample_indices(raw: str, fields) -> list[int]:
    if str(raw).strip().lower() != "all":
        indices = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
        return indices or [0]
    if not fields:
        raise RuntimeError("Cannot resolve --sample-indices all without at least one selected field.")
    return list(range(int(fields[0].shape[0])))


class EvaluationProgressReporter:
    def __init__(self, total_samples: int, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._last_refresh_at = 0.0
        self._refresh_interval = 0.2
        self.bar = tqdm(
            total=total_samples,
            desc="PDEBench eval",
            unit="sample",
            dynamic_ncols=True,
            leave=True,
            disable=not self.enabled,
        )

    def __call__(self, event: ProgressEvent) -> None:
        if not self.enabled:
            return
        self.bar.set_postfix_str(self._format_event(event), refresh=False)
        if event.phase == "sample_completed":
            self.bar.update(1)
            return
        if self._should_refresh(event.phase):
            self.bar.update(0)

    def close(self) -> None:
        self.bar.close()

    def _should_refresh(self, phase: str) -> bool:
        now = time.monotonic()
        if phase in {
            "sample_started",
            "sample_loaded",
            "reconstruction_started",
            "reconstruction_completed",
            "operator_started",
            "operator_completed",
            "hdf5_write_started",
            "hdf5_write_completed",
        }:
            self._last_refresh_at = now
            return True
        if now - self._last_refresh_at >= self._refresh_interval:
            self._last_refresh_at = now
            return True
        return False

    def _format_event(self, event: ProgressEvent) -> str:
        sample_label = self._format_sample_label(event)
        if event.phase == "sample_started":
            return f"{sample_label} loading"
        if event.phase == "sample_loaded":
            return f"{sample_label} loaded"
        if event.phase == "reconstruction_started":
            total = event.total_steps or 0
            return f"{sample_label} reconstruct 0/{total} frames"
        if event.phase == "reconstruction_batch_completed":
            done = event.step or 0
            total = event.total_steps or 0
            batch_index = event.batch_index or 0
            total_batches = event.total_batches or 0
            return (
                f"{sample_label} reconstruct {done}/{total} frames "
                f"(batch {batch_index}/{total_batches})"
            )
        if event.phase == "reconstruction_completed":
            total = event.total_steps or 0
            return f"{sample_label} reconstruction done ({total} frames)"
        if event.phase == "operator_started":
            return f"{sample_label} {event.operator_name}/{event.variant}"
        if event.phase == "operator_rollout_step":
            step = event.step or 0
            total = event.total_steps or 0
            return f"{sample_label} {event.operator_name}/{event.variant} rollout {step}/{total}"
        if event.phase == "operator_completed":
            return f"{sample_label} {event.operator_name}/{event.variant} done"
        if event.phase == "hdf5_write_started":
            return f"{sample_label} writing reconstructed HDF5"
        if event.phase == "hdf5_write_completed":
            return f"{sample_label} reconstructed HDF5 updated"
        if event.phase == "sample_completed":
            return f"{sample_label} completed"
        return sample_label

    @staticmethod
    def _format_sample_label(event: ProgressEvent) -> str:
        if event.sample_position is not None and event.sample_total is not None:
            return f"sample {event.sample_position}/{event.sample_total} (idx={event.sample_index})"
        if event.sample_index is not None:
            return f"sample idx={event.sample_index}"
        return "sample"


def evaluate_record_stream(
    records,
    operators: dict,
    reconstructed_hdf5_output,
    field_keys,
    args,
    time_slice,
    progress_callback=None,
):
    samples = []
    aggregate: dict[str, list[float]] = {}
    if reconstructed_hdf5_output is None:
        for record in records:
            sample_payload = evaluate_records(
                [record],
                operators,
                progress_callback=progress_callback,
            )["samples"][0]
            samples.append(sample_payload)
            for key, value in sample_payload["metrics"].items():
                if math.isfinite(float(value)):
                    aggregate.setdefault(key, []).append(float(value))
            emit_progress(
                progress_callback,
                "sample_completed",
                sample_index=record.sample_index,
            )
    else:
        with h5py.File(reconstructed_hdf5_output, "r+") as target:
            for record in records:
                emit_progress(
                    progress_callback,
                    "hdf5_write_started",
                    sample_index=record.sample_index,
                )
                write_reconstructed_record_to_hdf5(
                    target=target,
                    record=record,
                    field_keys=field_keys,
                    time_slice=time_slice,
                    spatial_stride=args.spatial_stride,
                )
                emit_progress(
                    progress_callback,
                    "hdf5_write_completed",
                    sample_index=record.sample_index,
                )
                sample_payload = evaluate_records(
                    [record],
                    operators,
                    progress_callback=progress_callback,
                )["samples"][0]
                samples.append(sample_payload)
                for key, value in sample_payload["metrics"].items():
                    if math.isfinite(float(value)):
                        aggregate.setdefault(key, []).append(float(value))
                emit_progress(
                    progress_callback,
                    "sample_completed",
                    sample_index=record.sample_index,
                )

    return {
        "samples": samples,
        "summary": summarize_metrics(aggregate),
    }


def main() -> None:
    args = parse_args()
    cli_field_keys = parse_fields(args.fields)
    discovered_fields = inspect_pdebench_fields(args.hdf5_path)
    discovered_field_keys = [field.path for field in discovered_fields]

    if not discovered_field_keys:
        raise RuntimeError(f"No compressible PDEBench fields found in {args.hdf5_path}.")

    reconstructor = None
    checkpoint_field_keys = None
    if args.compressor_checkpoint:
        reconstructor = CheckpointReconstructor(
            checkpoint_path=args.compressor_checkpoint,
            config_path=args.compressor_config,
            project_root=PROJECT_ROOT,
            device=args.device,
        )
        checkpoint_field_keys = resolve_checkpoint_field_keys(reconstructor.config)
        validate_checkpoint_field_keys_against_model(reconstructor.config, checkpoint_field_keys)

    field_keys = resolve_field_keys_for_evaluation(
        cli_field_keys=cli_field_keys,
        checkpoint_field_keys=checkpoint_field_keys,
        discovered_field_keys=discovered_field_keys,
    )
    fields = inspect_pdebench_fields(args.hdf5_path, field_keys=field_keys)

    operators = {}
    if args.forward_operator_type != "none":
        operators["forward"] = build_operator(
            args.forward_operator_type,
            spec=args.forward_operator_spec,
            checkpoint_path=args.forward_checkpoint,
            pdebench_root=args.pdebench_root,
            num_channels=args.num_channels or len(field_keys),
            initial_step=args.initial_step,
            t_train=args.t_train,
            modes=args.modes,
            width=args.width,
            init_features=args.init_features,
            device=args.device,
        )
    if args.inverse_operator_type != "none":
        operators["inverse"] = build_operator(
            args.inverse_operator_type,
            spec=args.inverse_operator_spec,
            device=args.device,
        )

    reconstructed_hdf5_output = None
    if args.reconstructed_hdf5_output:
        reconstructed_hdf5_output = prepare_reconstructed_hdf5_output(
            hdf5_path=args.hdf5_path,
            output_path=args.reconstructed_hdf5_output,
            overwrite=args.overwrite_reconstructed_hdf5,
        )

    sample_indices = parse_sample_indices(args.sample_indices, fields)
    time_slice = parse_time_slice(args.time_start, args.time_stop, args.time_step)
    operator_names = sorted(operators) if operators else ["reconstruction-only"]
    print(
        "Starting PDEBench downstream evaluation: "
        f"samples={len(sample_indices)}, fields={field_keys}, operators={operator_names}, "
        f"batch_size={args.batch_size}, spatial_stride={args.spatial_stride}"
    )
    progress_reporter = EvaluationProgressReporter(
        total_samples=len(sample_indices),
        enabled=not args.no_progress,
    )
    try:
        records_iter = generate_pdebench_records(
            hdf5_path=args.hdf5_path,
            field_keys=field_keys,
            sample_indices=sample_indices,
            reconstructor=reconstructor,
            batch_size=args.batch_size,
            time_slice=time_slice,
            spatial_stride=args.spatial_stride,
            progress_callback=progress_reporter,
        )
        results = evaluate_record_stream(
            records=records_iter,
            operators=operators,
            reconstructed_hdf5_output=reconstructed_hdf5_output,
            field_keys=field_keys,
            args=args,
            time_slice=time_slice,
            progress_callback=progress_reporter,
        )
    finally:
        progress_reporter.close()
    results["metadata"] = {
        "hdf5_path": str(Path(args.hdf5_path).expanduser().resolve()),
        "fields": [field.__dict__ for field in fields],
        "sample_indices": sample_indices,
        "operator_names": sorted(operators),
        "compressor_checkpoint": args.compressor_checkpoint,
        "reconstructed_hdf5_output": str(reconstructed_hdf5_output) if reconstructed_hdf5_output else None,
    }

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT
        / "outputs"
        / "pdebench_downstream"
        / f"{time.strftime('%Y%m%d_%H%M%S')}_pdebench_downstream.json"
    )
    dump_json(output_path, results)
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))
    print(f"Saved full result to: {output_path}")
    if reconstructed_hdf5_output:
        print(f"Saved reconstructed HDF5 to: {reconstructed_hdf5_output}")


if __name__ == "__main__":
    main()
