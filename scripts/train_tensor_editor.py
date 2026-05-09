from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.config import load_config
from tensor_compression.engine.tensor_editor_trainer import TensorEditorTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Chinese-prompt conditioned tensor restoration editor."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the tensor editor YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets/model and write setup_summary.json without training.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="Override editor.data.jsonl_path.",
    )
    parser.add_argument(
        "--compressor-checkpoint",
        type=str,
        default=None,
        help="Override editor.compressor.checkpoint_path.",
    )
    parser.add_argument(
        "--compressor-config",
        type=str,
        default=None,
        help="Override editor.compressor.config_path when the checkpoint lacks an embedded config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override experiment.device, e.g. cuda:0, cuda:7, cpu, or auto.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override experiment.output_root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override editor.data.loader.batch_size.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help="Override editor.data.validation_ratio.",
    )
    return parser.parse_args()


def _resolve_optional_path(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.jsonl_path is not None:
        config["editor"]["data"]["jsonl_path"] = _resolve_optional_path(args.jsonl_path)
    if args.compressor_checkpoint is not None:
        config["editor"]["compressor"]["checkpoint_path"] = _resolve_optional_path(
            args.compressor_checkpoint
        )
    if args.compressor_config is not None:
        config["editor"]["compressor"]["config_path"] = _resolve_optional_path(
            args.compressor_config
        )
    if args.device is not None:
        config["experiment"]["device"] = args.device
    if args.output_root is not None:
        config["experiment"]["output_root"] = _resolve_optional_path(args.output_root)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["editor"]["data"]["loader"]["batch_size"] = int(args.batch_size)
    if args.validation_ratio is not None:
        config["editor"]["data"]["validation_ratio"] = float(args.validation_ratio)
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config, base_root=PROJECT_ROOT)
    config = apply_overrides(config, args)
    trainer = TensorEditorTrainer(config=config, project_root=PROJECT_ROOT)
    if args.dry_run:
        trainer.validate_setup()
        print("Tensor editor dry run finished successfully.")
        return
    trainer.fit()


if __name__ == "__main__":
    main()
