from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {config_path}.")
    return payload


def nested_get(config: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def first_nested(config: Mapping[str, Any], dotted_paths: Sequence[str], default: Any = None) -> Any:
    for dotted_path in dotted_paths:
        value = nested_get(config, dotted_path, default=None)
        if value is not None:
            return value
    return default


def environment_override(name: str, configured_value: Any) -> Any:
    """Use a non-empty machine-local environment value before a portable config value."""

    override = os.environ.get(name)
    if override is None or not override.strip():
        return configured_value
    return override


def resolve_path_string(value: Any, project_root: str | Path) -> str | None:
    if value is None:
        return None
    raw = os.path.expandvars(str(value))
    path = Path(raw).expanduser()
    if raw.startswith("./") or raw.startswith("../"):
        path = Path(project_root) / path
    return str(path)


def value_to_csv(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return ",".join(str(item) for item in value)
    return str(value)


def set_default(args: Any, attr: str, value: Any, default: Any = None) -> None:
    if getattr(args, attr, None) is None:
        setattr(args, attr, value if value is not None else default)


def require_args(args: Any, required: Sequence[str]) -> None:
    missing = [name for name in required if getattr(args, name, None) in {None, ""}]
    if missing:
        formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Missing required argument(s): {formatted}. Pass them directly or through --config.")
