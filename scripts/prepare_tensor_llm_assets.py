from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
)


@dataclass(frozen=True)
class DiskInfo:
    path: Path
    total_gb: float
    used_gb: float
    free_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect candidate storage roots, create tensor-LLM asset directories, "
            "and optionally download the configured HuggingFace model."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/tensor_llm_adapter_pipeline.yaml",
        help="Tensor-LLM pipeline YAML config.",
    )
    parser.add_argument("--create-dirs", action="store_true", help="Create configured asset directories.")
    parser.add_argument("--download-model", action="store_true", help="Download the configured HF model.")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional HuggingFace token. If omitted, huggingface_hub uses its normal login/cache.",
    )
    return parser.parse_args()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def unique_paths(paths: list[str]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for raw in paths:
        resolved = str(Path(os.path.expandvars(raw)).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(Path(resolved))
    return unique


def inspect_disk(path: Path) -> DiskInfo | None:
    if not path.exists():
        return None
    usage = shutil.disk_usage(path)
    gb = 1024**3
    return DiskInfo(
        path=path,
        total_gb=usage.total / gb,
        used_gb=usage.used / gb,
        free_gb=usage.free / gb,
    )


def collect_disk_infos(config: dict[str, Any]) -> list[DiskInfo]:
    configured = as_list(first_nested(config, ["storage.candidate_roots"]))
    defaults = ["/data", "/scratch", "/mnt", "/home", str(PROJECT_ROOT)]
    infos = []
    for path in unique_paths(configured + defaults):
        info = inspect_disk(path)
        if info is not None:
            infos.append(info)
    return sorted(infos, key=lambda item: item.free_gb, reverse=True)


def print_disk_table(infos: list[DiskInfo], min_free_gb: float) -> None:
    print("Candidate storage roots:")
    if not infos:
        print("- no existing candidate roots found")
        return
    for info in infos:
        mark = "OK" if info.free_gb >= min_free_gb else "LOW"
        print(
            f"- {mark:3s} {str(info.path):32s} "
            f"free={info.free_gb:8.1f} GB used={info.used_gb:8.1f} GB total={info.total_gb:8.1f} GB"
        )


def configured_paths(config: dict[str, Any]) -> dict[str, str]:
    path_specs = {
        "asset_root": ["storage.asset_root"],
        "hf_home": ["storage.hf_home"],
        "model_local_dir": ["model.local_dir"],
        "qa_dir": ["data.qa_dir"],
        "latent_dir": ["data.latent_dir", "latent_export.output_dir"],
        "output_root": ["llm_training.output_root", "storage.output_root"],
    }
    paths: dict[str, str] = {}
    for name, candidates in path_specs.items():
        value = first_nested(config, candidates)
        if value is not None:
            paths[name] = str(resolve_path_string(value, PROJECT_ROOT))
    return paths


def create_directories(paths: dict[str, str]) -> None:
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if name == "model_local_dir":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
        print(f"created/exists: {name} -> {path}")


def write_env_file(paths: dict[str, str]) -> Path | None:
    asset_root = paths.get("asset_root")
    hf_home = paths.get("hf_home")
    if not asset_root or not hf_home:
        return None
    env_path = Path(asset_root) / "env_tensor_llm.sh"
    hub_cache = Path(hf_home) / "hub"
    lines = [
        f"export HF_HOME={hf_home}",
        f"export HUGGINGFACE_HUB_CACHE={hub_cache}",
        f"export TRANSFORMERS_CACHE={hf_home}",
    ]
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def download_model(config: dict[str, Any], paths: dict[str, str], token: str | None) -> str:
    model_name = first_nested(config, ["model.name_or_path", "model.model_name_or_path"])
    if not model_name:
        raise ValueError("Config must set model.name_or_path before --download-model can be used.")
    model_path = Path(str(model_name)).expanduser()
    if model_path.exists():
        print(f"Model already points to a local path: {model_path}")
        return str(model_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - missing dependency path
        raise ImportError("Install huggingface_hub or transformers before downloading HF models.") from exc

    kwargs: dict[str, Any] = {
        "repo_id": str(model_name),
        "revision": str(first_nested(config, ["model.revision"], default="main")),
        "cache_dir": paths.get("hf_home"),
        "token": token,
    }
    if paths.get("model_local_dir"):
        kwargs["local_dir"] = paths["model_local_dir"]
    allow_patterns = first_nested(config, ["model.allow_patterns"])
    ignore_patterns = first_nested(config, ["model.ignore_patterns"])
    if allow_patterns:
        kwargs["allow_patterns"] = as_list(allow_patterns)
    if ignore_patterns:
        kwargs["ignore_patterns"] = as_list(ignore_patterns)

    target = snapshot_download(**kwargs)
    print(f"Downloaded/resolved model: {target}")
    return str(target)


def print_next_commands(config_path: Path, paths: dict[str, str]) -> None:
    print()
    print("Next commands:")
    if paths.get("asset_root"):
        print(f"source {Path(paths['asset_root']) / 'env_tensor_llm.sh'}")
    print(f"python scripts/export_tensor_readout_latents.py --config {config_path}")
    print(f"CUDA_VISIBLE_DEVICES=1 python scripts/train_tensor_llm_adapter.py --config {config_path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = load_yaml_mapping(config_path)
    min_free_gb = float(first_nested(config, ["storage.min_free_gb"], default=100.0))
    infos = collect_disk_infos(config)
    print_disk_table(infos, min_free_gb=min_free_gb)

    paths = configured_paths(config)
    print()
    print("Configured paths:")
    print(json.dumps(paths, indent=2, ensure_ascii=False))

    if args.create_dirs:
        print()
        create_directories(paths)
        env_path = write_env_file(paths)
        if env_path is not None:
            print(f"wrote env file: {env_path}")

    if args.download_model:
        print()
        downloaded = download_model(config, paths, token=args.token)
        print(f"model_path={downloaded}")

    print_next_commands(config_path, paths)


if __name__ == "__main__":
    main()
