from __future__ import annotations

from tensor_compression.registry import Registry

EDITOR_REGISTRY = Registry("editors")

from .conditional_tensor_editor_2d import ConditionalTensorEditor2D  # noqa: E402,F401

__all__ = ["EDITOR_REGISTRY", "ConditionalTensorEditor2D"]
