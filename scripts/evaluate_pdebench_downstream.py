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
    build_operator,
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
    return parser.parse_args()


def parse_sample_indices(raw: str, fields) -> list[int]:
    if str(raw).strip().lower() != "all":
        indices = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
        return indices or [0]
    if not fields:
        raise RuntimeError("Cannot resolve --sample-indices all without at least one selected field.")
    return list(range(int(fields[0].shape[0])))


def evaluate_record_stream(records, operators: dict, reconstructed_hdf5_output, field_keys, args, time_slice):
    samples = []
    aggregate: dict[str, list[float]] = {}
    if reconstructed_hdf5_output is None:
        for record in records:
            sample_payload = evaluate_records([record], operators)["samples"][0]
            samples.append(sample_payload)
            for key, value in sample_payload["metrics"].items():
                if math.isfinite(float(value)):
                    aggregate.setdefault(key, []).append(float(value))
    else:
        with h5py.File(reconstructed_hdf5_output, "r+") as target:
            for record in records:
                write_reconstructed_record_to_hdf5(
                    target=target,
                    record=record,
                    field_keys=field_keys,
                    time_slice=time_slice,
                    spatial_stride=args.spatial_stride,
                )
                sample_payload = evaluate_records([record], operators)["samples"][0]
                samples.append(sample_payload)
                for key, value in sample_payload["metrics"].items():
                    if math.isfinite(float(value)):
                        aggregate.setdefault(key, []).append(float(value))

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
    records_iter = generate_pdebench_records(
        hdf5_path=args.hdf5_path,
        field_keys=field_keys,
        sample_indices=sample_indices,
        reconstructor=reconstructor,
        batch_size=args.batch_size,
        time_slice=time_slice,
        spatial_stride=args.spatial_stride,
    )
    results = evaluate_record_stream(
        records=records_iter,
        operators=operators,
        reconstructed_hdf5_output=reconstructed_hdf5_output,
        field_keys=field_keys,
        args=args,
        time_slice=time_slice,
    )
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
