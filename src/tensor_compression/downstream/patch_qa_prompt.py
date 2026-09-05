from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def task_specific_instruction(record: Mapping[str, Any]) -> str:
    task_type = str(record.get("task_type", "")).strip()
    if task_type == "normalized_point_value":
        return (
            "Rule: read the standardized value z directly at the requested patch-local row and column from "
            "the tensor soft tokens. "
            "Choose the closest numeric option and return only its label."
        )
    if task_type == "raw_point_value_with_stats":
        return (
            "Rule: read standardized z at the requested patch-local position from the tensor soft tokens, then "
            "recover the original value with x = mean + scale * z using the stated patch statistics. "
            "Choose the closest original-value option and return only its label."
        )
    if task_type == "point_bin":
        return (
            "Rule: read the requested field value at the given row and col from the tensor soft tokens. "
            "Return its quantile-bin label. Bin labels B00,B01,... are ordered from low to high: "
            "B00 means the lowest value range, larger bin numbers mean larger value ranges, and the last bin "
            "means the highest value range. Return exactly one listed label and no extra text."
        )
    if task_type == "point_compare":
        return (
            "Rule: compare the standardized values at point A and point B using the tensor soft tokens; "
            "per-patch standardization preserves their original ordering. "
            "Choice A means point A is greater than or tied with point B. "
            "Choice B means point B is strictly greater than point A. "
            "Return exactly A or B and no extra text."
        )
    if task_type == "patch_compare":
        return (
            "Rule: compare the mean requested field value over patch A with patch B using the tensor soft tokens. "
            "Choice A means patch A has greater or tied mean. "
            "Choice B means patch B has strictly greater mean. "
            "Return exactly A or B and no extra text."
        )
    if task_type == "region_mean_compare":
        return (
            "Rule: compare the standardized means in the two stated patch-local regions using the tensor soft "
            "tokens; per-patch standardization preserves their original ordering. "
            "Return A if region A has the greater or tied mean; otherwise return B."
        )
    if task_type == "extreme_quadrant":
        return (
            "Rule: locate the requested maximum or minimum in the standardized patch using the tensor soft "
            "tokens; per-patch standardization preserves extrema and their locations. "
            "Return A for top-left, B for top-right, C for bottom-left, or D for bottom-right."
        )
    if task_type == "max_speed_quadrant":
        return (
            "Rule: find the grid cell with maximum speed magnitude from the tensor soft tokens. "
            "Return the quadrant label of that cell. Return exactly one listed label and no extra text."
        )
    if task_type == "global_stat_bin":
        return (
            "Rule: compute the requested global speed statistic from the tensor soft tokens. "
            "Return its quantile-bin label. Bin labels B00,B01,... are ordered from low to high: "
            "B00 means the lowest value range, larger bin numbers mean larger value ranges, and the last bin "
            "means the highest value range. Return exactly one listed label and no extra text."
        )
    return (
        "Rule: answer the tensor readout query using the tensor soft tokens. "
        "Return exactly one listed label and no extra text."
    )


def choice_semantics(record: Mapping[str, Any]) -> str:
    task_type = str(record.get("task_type", "")).strip()
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        choices = []
    labels = [str(choice) for choice in choices]
    if task_type in {"point_bin", "global_stat_bin"} and labels:
        return (
            "Choice meanings: "
            + "; ".join(
                f"{label}=quantile bin {index} of {len(labels) - 1}, ordered from low to high"
                for index, label in enumerate(labels)
            )
            + "."
        )
    if task_type == "point_compare":
        return "Choice meanings: A=point A is greater than or tied with point B; B=point B is strictly greater than point A."
    if task_type == "patch_compare":
        return "Choice meanings: A=patch A mean is greater than or tied with patch B mean; B=patch B mean is strictly greater than patch A mean."
    if task_type == "region_mean_compare":
        return "Choice meanings: A=region A has greater or tied mean; B=region B has strictly greater mean."
    if task_type == "extreme_quadrant":
        return "Choice meanings: A=top-left; B=top-right; C=bottom-left; D=bottom-right."
    if task_type == "max_speed_quadrant":
        return "Choice meanings: quadrant labels refer to the location of the maximum-speed grid cell."
    if labels:
        return "Choice meanings: choose exactly one of the listed labels."
    return ""


def valid_choice_instruction(record: Mapping[str, Any]) -> str:
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        raise ValueError(f"Record has no valid choices: {record.get('qa_id', '<unknown>')}")
    labels = [str(choice) for choice in choices]
    return (
        f"Required output: exactly one of {', '.join(labels)}. "
        "Output only that label, with no explanation, punctuation, or other text."
    )


def build_prompt(record: Mapping[str, Any], prompt_template: str) -> str:
    query = str(record.get("query") or record.get("question") or "")
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        choices = []
    choice_text = ", ".join(str(choice) for choice in choices)
    if prompt_template == "field_memory":
        return (
            "The supplied field memory represents the numerical grid, with one memory cell per grid cell.\n"
            "Rows and columns in the query use one-based indices. Use the field values to answer.\n\n"
            f"Query: {query}\n"
            f"Choices: {choice_text}\n"
            f"{choice_semantics(record)}\n"
            f"{valid_choice_instruction(record)}\n"
            "Answer:"
        )
    if prompt_template == "generic":
        return (
            "Tensor-state soft tokens are prepended before this text.\n"
            "Answer the tensor readout query using those tokens.\n\n"
            f"Query: {query}\n"
            f"Choices: {choice_text}\n"
            f"{valid_choice_instruction(record)}\n"
            "Answer:"
        )
    if prompt_template != "task_specific":
        raise ValueError(f"Unsupported prompt template: {prompt_template}")
    return (
        "Tensor soft tokens before this text encode the tensor state.\n"
        f"{task_specific_instruction(record)}\n"
        "Do not answer from coordinate or label priors alone; use the tensor soft tokens for numeric values.\n\n"
        f"Query: {query}\n"
        f"Choices: {choice_text}\n"
        f"{choice_semantics(record)}\n"
        f"{valid_choice_instruction(record)}\n"
        "Answer:"
    )
