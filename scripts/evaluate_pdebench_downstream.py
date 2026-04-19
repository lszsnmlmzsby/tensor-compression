from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import (
    CheckpointReconstructor,
    build_operator,
    evaluate_records,
    inspect_pdebench_fields,
    iter_pdebench_records,
    parse_fields,
    parse_indices,
    parse_time_slice,
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
        help="Comma-separated HDF5 field keys. Defaults to discoverable PDE fields.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0",
        help="Comma-separated sample indices to evaluate, e.g. 0,1,2.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    field_keys = parse_fields(args.fields)
    if field_keys is None:
        fields = inspect_pdebench_fields(args.hdf5_path)
        field_keys = [field.path for field in fields]
    else:
        fields = inspect_pdebench_fields(args.hdf5_path, field_keys=field_keys)

    if not field_keys:
        raise RuntimeError(f"No compressible PDEBench fields found in {args.hdf5_path}.")

    reconstructor = None
    if args.compressor_checkpoint:
        reconstructor = CheckpointReconstructor(
            checkpoint_path=args.compressor_checkpoint,
            config_path=args.compressor_config,
            project_root=PROJECT_ROOT,
            device=args.device,
        )

    records = iter_pdebench_records(
        hdf5_path=args.hdf5_path,
        field_keys=field_keys,
        sample_indices=parse_indices(args.sample_indices),
        reconstructor=reconstructor,
        batch_size=args.batch_size,
        time_slice=parse_time_slice(args.time_start, args.time_stop, args.time_step),
        spatial_stride=args.spatial_stride,
    )

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

    results = evaluate_records(records, operators)
    results["metadata"] = {
        "hdf5_path": str(Path(args.hdf5_path).expanduser().resolve()),
        "fields": [field.__dict__ for field in fields],
        "sample_indices": [record.sample_index for record in records],
        "operator_names": sorted(operators),
        "compressor_checkpoint": args.compressor_checkpoint,
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


if __name__ == "__main__":
    main()
