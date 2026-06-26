from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_tensor_llm_adapter.py"
DIAGNOSE_SCRIPT = PROJECT_ROOT / "scripts" / "diagnose_tensor_llm_adapter.py"


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
    parser.add_argument(
        "--diagnose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run diagnose_tensor_llm_adapter.py on adapter_best.pt after a successful overfit run.",
    )
    parser.add_argument(
        "--diagnose-records",
        type=int,
        default=64,
        help="Number of records written by the post-training diagnostic pass.",
    )
    parser.add_argument(
        "--diagnose-hidden-records",
        type=int,
        default=16,
        help="Number of diagnostic records that include LLM hidden-state summaries.",
    )
    parser.add_argument(
        "--diagnose-split",
        type=str,
        default=None,
        help="Split used for diagnostics. Defaults to --source-split.",
    )
    args, passthrough = parser.parse_known_args()
    if args.records <= 0:
        raise ValueError("--records must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.diagnose_records <= 0:
        raise ValueError("--diagnose-records must be positive.")
    if args.diagnose_hidden_records < 0:
        raise ValueError("--diagnose-hidden-records must be non-negative.")
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def read_configured_output_root(config_path: str) -> Path:
    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return PROJECT_ROOT / "outputs" / "tensor_llm_outputs" / "runs"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        return PROJECT_ROOT / "outputs" / "tensor_llm_outputs" / "runs"
    llm_training = config.get("llm_training") if isinstance(config.get("llm_training"), dict) else {}
    storage = config.get("storage") if isinstance(config.get("storage"), dict) else {}
    raw = llm_training.get("output_root") or storage.get("output_root") or "./outputs/tensor_llm_outputs/runs"
    output_root = Path(str(raw)).expanduser()
    if str(raw).startswith("./") or str(raw).startswith("../"):
        output_root = PROJECT_ROOT / output_root
    return output_root


def snapshot_run_dirs(output_root: Path) -> set[Path]:
    if not output_root.exists():
        return set()
    return {path.resolve() for path in output_root.iterdir() if path.is_dir()}


def find_created_run_dir(output_root: Path, before: set[Path], run_name: str) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.resolve() not in before and path.name.endswith(f"_{run_name}")
    ]
    if not candidates:
        candidates = [path for path in output_root.iterdir() if path.is_dir() and path.name.endswith(f"_{run_name}")]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_diagnose_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    checkpoint = run_dir / "adapter_best.pt"
    output = run_dir / f"adapter_best_diagnostics_{args.diagnose_split or args.source_split}.jsonl"
    return [
        sys.executable,
        str(DIAGNOSE_SCRIPT),
        "--config",
        str(args.config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        str(args.diagnose_split or args.source_split),
        "--records",
        str(args.diagnose_records),
        "--max-choice-records",
        str(args.diagnose_hidden_records),
        "--output",
        str(output),
    ]


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
    output_root = read_configured_output_root(str(args.config))
    before = snapshot_run_dirs(output_root)
    if args.dry_run:
        print("Post-training diagnostics:")
        print("enabled" if args.diagnose else "disabled")
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    run_dir = find_created_run_dir(output_root, before, str(args.run_name))
    if run_dir is None:
        print(
            json.dumps(
                {
                    "warning": "Training finished, but the overfit wrapper could not locate the run directory.",
                    "output_root": str(output_root),
                    "run_name": str(args.run_name),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(0)
    print(json.dumps({"run_dir": str(run_dir)}, indent=2, ensure_ascii=False))
    if args.diagnose:
        diagnose_command = build_diagnose_command(args, run_dir)
        print("Post-training diagnostic command:")
        print(shlex.join(diagnose_command))
        diagnose_completed = subprocess.run(diagnose_command, cwd=PROJECT_ROOT)
        raise SystemExit(diagnose_completed.returncode)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
