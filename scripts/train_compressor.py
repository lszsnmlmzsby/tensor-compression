from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.config import load_config
from tensor_compression.engine.trainer import CompressionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tensor compressor.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build objects and validate config without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, base_root=PROJECT_ROOT)
    trainer = CompressionTrainer(config=config, project_root=PROJECT_ROOT)
    if args.dry_run:
        trainer.validate_setup()
        print("Dry run finished successfully.")
        return
    trainer.fit()


if __name__ == "__main__":
    main()
