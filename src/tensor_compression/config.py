from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _resolve_value(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, root) for item in value]
    if isinstance(value, str):
        if value.startswith("./"):
            return str((root / value[2:]).resolve())
        return value
    return value


def load_config(path: str | Path, base_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    resolution_root = Path(base_root).resolve() if base_root is not None else Path.cwd().resolve()
    resolved = _resolve_value(copy.deepcopy(config), resolution_root)
    resolved["config_path"] = str(config_path)
    resolved["project_root"] = str(resolution_root)
    return resolved
