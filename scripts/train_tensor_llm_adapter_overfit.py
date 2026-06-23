from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_tensor_llm_adapter.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tensor-LLM adapter overfit sanity check. "
            "The same QA split and first N records are used for train, val, and test, "
            "so correct-latent accuracy should rise above no_latent/shuffled if the "
            "adapter can learn to depend on tensor latents at all."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tensor_llm_adapter_pipeline.yaml",
        help="Adapter pipeline config passed to train_tensor_llm_adapter.py.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="train",
        help="QA split used for train, val, and test in the overfit check.",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=2048,
        help="Number of records from --source-split used for train, val, and test.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of overfit epochs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="tensor_llm_adapter_overfit",
        help="Run name override for the overfit check.",
    )
    parser.add_argument(
        "--eval-baselines",
        type=str,
        default="correct,no_latent,shuffled",
        help="Baselines used to test whether correct latents matter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved train command without executing it.",
    )
    args, passthrough = parser.parse_known_args()
    if args.records <= 0:
        raise ValueError("--records must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(args.config),
        "--run-name",
        str(args.run_name),
        "--train-split",
        str(args.source_split),
        "--val-split",
        str(args.source_split),
        "--test-split",
        str(args.source_split),
        "--max-train-records",
        str(args.records),
        "--max-val-records",
        str(args.records),
        "--max-test-records",
        str(args.records),
        "--epochs",
        str(args.epochs),
        "--eval-baselines",
        str(args.eval_baselines),
    ]
    command.extend(passthrough)
    return command


def main() -> None:
    args, passthrough = parse_args()
    command = build_command(args, passthrough)
    print("Overfit sanity check command:")
    print(shlex.join(command))
    if args.dry_run:
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
