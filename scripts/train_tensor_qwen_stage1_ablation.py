from __future__ import annotations

"""Run a Stage-1 alignment ablation through the unchanged production trainers.

The ablation has two phases:

1. ``direct`` trains the normal direct-QA spatial adapter after replacing only
   the learned Stage-1 alignment adapter state with a deterministic random state.
2. ``dense`` trains the normal dense cross-attention model from the resulting
   direct-QA checkpoint.

The cached latents, frozen field encoder, adapter architecture, Qwen model,
datasets, losses, and downstream training budgets are preserved.  The source
Stage-1 checkpoint remains untouched and continues to satisfy the immutable
latent provenance checks.  All wrapping is process-local; neither production
trainer is modified.
"""

import argparse
import copy
import hashlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402
from tensor_compression.downstream.patch_qa_contract import sha256_file  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    load_yaml_mapping,
    resolve_path_string,
)


ABLATION_CONTRACT_VERSION = 1
ABLATION_MODE = "random_spatial_adapter_preserve_latent_encoder"
DISABLED_PATH_VALUES = {"", "none", "null", "random"}


def _resolved(value: Any, *, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing required path: {label}.")
    return Path(resolve_path_string(value, PROJECT_ROOT)).expanduser().resolve()


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _disabled_path(value: Any) -> bool:
    return str(value or "").strip().lower() in DISABLED_PATH_VALUES


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError("Cannot hash an adapter state with no tensors.")
    return digest.hexdigest()


def _checkpoint_adapter_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("adapter_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Stage-1 checkpoint is missing adapter_state_dict.")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Stage-1 adapter_state_dict contains a non-tensor value.")
    return state


def _random_spatial_state(payload: Mapping[str, Any], seed: int) -> dict[str, torch.Tensor]:
    state = _checkpoint_adapter_state(payload)
    checkpoint_args = payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-1 checkpoint is missing its argument contract.")

    latent_weight = state.get("latent_projection.weight")
    output_weight = state.get("output.1.weight")
    position = state.get("spatial_pos_encoding")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (latent_weight, output_weight, position)
    ):
        raise ValueError(
            "Stage-1 ablation requires a spatial_transformer alignment checkpoint."
        )
    if position.ndim != 3:
        raise ValueError(
            f"Expected [1,tokens,dim] spatial_pos_encoding, got {tuple(position.shape)}."
        )

    token_count = int(position.shape[-2])
    side = int(round(token_count**0.5))
    if side * side != token_count:
        raise ValueError(f"Spatial token count must form a square grid, got {token_count}.")
    block_indices = {
        int(str(name).split(".")[1])
        for name in state
        if str(name).startswith("blocks.") and len(str(name).split(".")) > 2
    }
    layers = max(block_indices) + 1 if block_indices else int(
        checkpoint_args.get("adapter_layers", 2)
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=int(latent_weight.shape[1]),
            latent_grid=(side, side),
            adapter_dim=int(latent_weight.shape[0]),
            projection_dim=int(output_weight.shape[0]),
            dropout=float(checkpoint_args.get("dropout", 0.0)),
            adapter_type="spatial_transformer",
            query_tokens=token_count,
            adapter_layers=layers,
            adapter_heads=int(checkpoint_args.get("adapter_heads", 8)),
            soft_prompt_scale=float(checkpoint_args.get("soft_prompt_scale", 0.05)),
        )

    fresh = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    expected_keys = set(state)
    observed_keys = set(fresh)
    if observed_keys != expected_keys:
        raise ValueError(
            "Fresh adapter does not exactly match the Stage-1 architecture; "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}."
        )
    incompatible = {
        name: {
            "expected_shape": tuple(state[name].shape),
            "observed_shape": tuple(fresh[name].shape),
            "expected_dtype": str(state[name].dtype),
            "observed_dtype": str(fresh[name].dtype),
        }
        for name in sorted(expected_keys)
        if tuple(fresh[name].shape) != tuple(state[name].shape)
        or fresh[name].dtype != state[name].dtype
    }
    if incompatible:
        raise ValueError(f"Fresh adapter tensor contract differs from Stage 1: {incompatible}.")
    return fresh


def _build_ablation_contract(
    source: Path,
    source_payload: Mapping[str, Any],
    random_state: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    checkpoint_args = source_payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-1 checkpoint is missing its argument contract.")
    encoder_trained_during_alignment = bool(checkpoint_args.get("train_patch_ae", False)) and not bool(
        checkpoint_args.get("freeze_patch_ae_after_pretrain", True)
    )
    if encoder_trained_during_alignment:
        raise ValueError(
            "This single-factor ablation cannot preserve an encoder that was updated by "
            "Stage-1 alignment. Use the frozen-encoder Stage-1 checkpoint or define a "
            "separate joint encoder-and-adapter ablation."
        )
    source_state_sha256 = _state_digest(_checkpoint_adapter_state(source_payload))
    random_state_sha256 = _state_digest(random_state)
    if source_state_sha256 == random_state_sha256:
        raise ValueError(
            "The random adapter exactly matches the Stage-1 adapter; the requested factor "
            "was not removed. Choose another ablation seed or inspect the source checkpoint."
        )
    return {
        "contract_version": ABLATION_CONTRACT_VERSION,
        "enabled": True,
        "factor_removed": "stage1_tensor_text_alignment_pretraining",
        "mode": ABLATION_MODE,
        "seed": int(seed),
        "source_stage1_checkpoint": str(source),
        "source_stage1_sha256": sha256_file(source),
        "source_adapter_state_sha256": source_state_sha256,
        "random_adapter_state_sha256": random_state_sha256,
        "source_encoder_trained_during_alignment": False,
        "changed": ["initial_spatial_adapter_learned_state"],
        "preserved": [
            "cached_latents",
            "frozen_field_encoder",
            "spatial_adapter_architecture_and_parameter_count",
            "direct_qa_objective_data_and_configured_example_budget",
            "dense_cross_attention_objective_data_and_configured_example_budget",
            "frozen_qwen",
        ],
    }


def _load_source_ablation(
    source: Path,
    seed: int,
    load_fn: Any = torch.load,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    payload = load_fn(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 checkpoint payload must be a mapping.")
    payload_copy = dict(payload)
    random_state = _random_spatial_state(payload_copy, seed)
    contract = _build_ablation_contract(source, payload_copy, random_state, seed)
    return payload_copy, random_state, contract


def _override_cli_tokens(overrides: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for raw_name, value in overrides.items():
        name = str(raw_name).strip()
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
            raise ValueError(f"Invalid direct trainer argument name: {raw_name!r}.")
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            tokens.append(option if value else "--no-" + option[2:])
            continue
        if value is None:
            tokens.extend((option, "none"))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tokens.extend((option, ",".join(str(item) for item in value)))
            continue
        if isinstance(value, Mapping):
            raise ValueError(f"Direct trainer override {name!r} cannot be a nested mapping.")
        tokens.extend((option, str(value)))
    return tokens


def _validate_direct_args(args: argparse.Namespace, source: Path) -> None:
    if str(args.adapter_architecture) != "alignment_adapter":
        raise ValueError(
            "Stage-1 ablation direct phase must use adapter_architecture=alignment_adapter."
        )
    configured_source = Path(str(args.adapter_init_checkpoint)).expanduser().resolve()
    if configured_source != source:
        raise ValueError(
            "The direct adapter initializer must be the source checkpoint named by the "
            f"ablation contract: configured={configured_source}, source={source}."
        )
    if not _disabled_path(args.stage2_warm_start_checkpoint) or not _disabled_path(
        args.stage2b_resume_checkpoint
    ):
        raise ValueError(
            "Stage-1 ablation must train Direct QA from the randomized Stage-1 state; "
            "Stage-2 warm-start and resume checkpoints must be disabled."
        )
    active_modes = [
        name
        for name in (
            "joint_ab_training",
            "point_reader_training",
            "full_local_reader_training",
        )
        if bool(getattr(args, name, False))
    ]
    if active_modes:
        raise ValueError(
            "Direct Stage-1 ablation cannot enable later Stage-2B modes: "
            f"{active_modes}."
        )


def _validate_ablation_contract(
    contract: Mapping[str, Any],
    configured_source: Path,
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(contract))
    if int(normalized.get("contract_version", 0)) != ABLATION_CONTRACT_VERSION:
        raise ValueError("Unsupported or missing Stage-1 ablation contract version.")
    if normalized.get("enabled") is not True or normalized.get("mode") != ABLATION_MODE:
        raise ValueError("Checkpoint is not a Stage-1 alignment ablation checkpoint.")
    contract_source = normalized.get("source_stage1_checkpoint")
    if not contract_source:
        raise ValueError("Ablation checkpoint does not record its source Stage-1 path.")
    actual_source_sha256 = sha256_file(configured_source)
    if normalized.get("source_stage1_sha256") != actual_source_sha256:
        raise ValueError(
            "Source Stage-1 checkpoint changed after the ablation direct phase."
        )
    seed = int(normalized.get("seed", -1))
    source_payload, random_state, expected = _load_source_ablation(
        configured_source,
        seed,
    )
    del source_payload, random_state
    for key in (
        "source_adapter_state_sha256",
        "random_adapter_state_sha256",
        "source_encoder_trained_during_alignment",
        "factor_removed",
    ):
        if normalized.get(key) != expected.get(key):
            raise ValueError(
                f"Ablation contract field {key!r} cannot be reproduced: "
                f"observed={normalized.get(key)!r}, expected={expected.get(key)!r}."
            )
    return normalized


def _ablation_contract_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "contract_version",
        "enabled",
        "factor_removed",
        "mode",
        "seed",
        "source_stage1_sha256",
        "source_adapter_state_sha256",
        "random_adapter_state_sha256",
        "source_encoder_trained_during_alignment",
        "direct_qa_checkpoint_sha256",
    )
    return {key: contract.get(key) for key in keys if key in contract}


def _run_direct(config: Mapping[str, Any], forwarded: list[str]) -> None:
    import scripts.train_tensor_llm_adapter as trainer

    section = config.get("ablation", {})
    direct = config.get("direct", {})
    if not isinstance(section, Mapping) or not isinstance(direct, Mapping):
        raise ValueError("Ablation config requires mapping sections: ablation and direct.")
    source = _resolved(
        section.get("source_stage1_checkpoint"),
        label="ablation.source_stage1_checkpoint",
    )
    if not source.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {source}")
    seed = int(section.get("seed", 42))
    base_config = _resolved(direct.get("base_config"), label="direct.base_config")
    if not base_config.is_file():
        raise FileNotFoundError(f"Direct base config not found: {base_config}")
    overrides = direct.get("arg_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("direct.arg_overrides must be a mapping of trainer argument names.")
    override_tokens = _override_cli_tokens(overrides)

    original_parse = trainer.parse_args
    original_load = torch.load
    original_save = trainer.atomic_torch_save
    original_config_snapshot = trainer.redacted_config_snapshot
    original_argv = list(sys.argv)
    source_payload, random_state, contract = _load_source_ablation(
        source,
        seed,
        load_fn=original_load,
    )
    del source_payload

    def parse_args() -> argparse.Namespace:
        args = original_parse()
        _validate_direct_args(args, source)
        args.stage1_ablation = True
        args.stage1_ablation_mode = ABLATION_MODE
        args.stage1_ablation_seed = seed
        args.stage1_ablation_source_checkpoint = str(source)
        args.stage1_ablation_contract = copy.deepcopy(contract)
        return args

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            candidate = Path(path)
            matches = candidate.name == source.name and _same_path(candidate, source)
        except (TypeError, OSError, ValueError):
            matches = False
        payload = original_load(path, *args, **kwargs)
        if not matches:
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("Stage-1 checkpoint payload must be a mapping.")
        replaced = dict(payload)
        replaced["adapter_state_dict"] = random_state
        replaced["ablation_contract"] = copy.deepcopy(contract)
        return replaced

    def save(path: Any, payload: Mapping[str, Any]) -> None:
        enriched = dict(payload)
        if enriched.get("checkpoint_type") == "tensor_llm_adapter":
            enriched["ablation_contract"] = copy.deepcopy(contract)
        original_save(path, enriched)

    def config_snapshot(config_payload: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = original_config_snapshot(config_payload)
        snapshot["stage1_ablation"] = copy.deepcopy(contract)
        snapshot["stage1_ablation_direct_arg_overrides"] = copy.deepcopy(
            dict(overrides)
        )
        return snapshot

    trainer.parse_args = parse_args
    trainer.atomic_torch_save = save
    trainer.redacted_config_snapshot = config_snapshot
    torch.load = load
    sys.argv = [
        original_argv[0],
        "--config",
        str(base_config),
        *override_tokens,
        *forwarded,
    ]
    try:
        trainer.main()
    finally:
        trainer.parse_args = original_parse
        trainer.atomic_torch_save = original_save
        trainer.redacted_config_snapshot = original_config_snapshot
        torch.load = original_load
        sys.argv = original_argv


def _run_dense(
    config: Mapping[str, Any],
    init_checkpoint: str,
    forwarded: list[str],
) -> None:
    import scripts.train_tensor_qwen_cross_attention as trainer

    section = config.get("ablation", {})
    dense = config.get("dense", {})
    if not isinstance(section, Mapping) or not isinstance(dense, Mapping):
        raise ValueError("Ablation config requires mapping sections: ablation and dense.")
    source = _resolved(
        section.get("source_stage1_checkpoint"),
        label="ablation.source_stage1_checkpoint",
    )
    base_config = _resolved(dense.get("base_config"), label="dense.base_config")
    if not base_config.is_file():
        raise FileNotFoundError(f"Dense base config not found: {base_config}")
    init_path = Path(init_checkpoint).expanduser().resolve()
    if not init_path.is_file():
        raise FileNotFoundError(f"Ablated Direct-QA checkpoint not found: {init_path}")
    payload = torch.load(init_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Ablated Direct-QA checkpoint must be a mapping.")
    if payload.get("checkpoint_type") != "tensor_llm_adapter":
        raise ValueError("Dense phase requires a Direct-QA tensor_llm_adapter checkpoint.")
    raw_contract = payload.get("ablation_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError(
            "Dense Stage-1 ablation requires adapter_best.pt produced by this script's direct phase."
        )
    contract = _validate_ablation_contract(raw_contract, source)
    direct_args = payload.get("args")
    if not isinstance(direct_args, Mapping) or direct_args.get(
        "adapter_architecture"
    ) != "alignment_adapter":
        raise ValueError("Ablated Direct-QA checkpoint has an invalid adapter architecture.")
    dense_contract = {
        **contract,
        "runtime_source_stage1_checkpoint": str(source),
        "direct_qa_checkpoint": str(init_path),
        "direct_qa_checkpoint_sha256": sha256_file(init_path),
    }

    original_parse = trainer.parse_args
    original_architecture_contract = trainer.architecture_contract
    original_validate_checkpoint = trainer.validate_checkpoint_contract
    original_argv = list(sys.argv)

    def parse_args() -> argparse.Namespace:
        args = original_parse()
        args.memory_init_checkpoint = str(init_path)
        args.stage1_ablation = True
        args.stage1_ablation_mode = ABLATION_MODE
        args.stage1_ablation_contract = copy.deepcopy(dense_contract)
        raw = copy.deepcopy(args.raw_config)
        raw_memory = raw.get("memory")
        memory = copy.deepcopy(dict(raw_memory)) if isinstance(raw_memory, Mapping) else {}
        memory["init_checkpoint"] = str(init_path)
        raw["memory"] = memory
        raw["stage1_ablation"] = copy.deepcopy(dense_contract)
        args.raw_config = raw
        trainer.validate_args(args)
        return args

    def architecture_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        architecture = original_architecture_contract(*args, **kwargs)
        architecture["stage1_ablation"] = copy.deepcopy(dense_contract)
        return architecture

    def validate_checkpoint_contract(
        checkpoint: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> None:
        original_validate_checkpoint(checkpoint, expected)
        observed_architecture = checkpoint.get("architecture")
        observed = (
            observed_architecture.get("stage1_ablation")
            if isinstance(observed_architecture, Mapping)
            else None
        )
        expected_ablation = expected.get("stage1_ablation")
        if not isinstance(observed, Mapping) or not isinstance(
            expected_ablation, Mapping
        ) or _ablation_contract_identity(observed) != _ablation_contract_identity(
            expected_ablation
        ):
            raise ValueError(
                "Dense checkpoint does not belong to this Stage-1 ablation lineage."
            )

    trainer.parse_args = parse_args
    trainer.architecture_contract = architecture_contract
    trainer.validate_checkpoint_contract = validate_checkpoint_contract
    sys.argv = [original_argv[0], "--config", str(base_config), *forwarded]
    try:
        trainer.main()
    finally:
        trainer.parse_args = original_parse
        trainer.architecture_contract = original_architecture_contract
        trainer.validate_checkpoint_contract = original_validate_checkpoint
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("direct", "dense"))
    parser.add_argument("--config", required=True, help="Stage-1 ablation orchestration YAML.")
    parser.add_argument(
        "--spatial-init-checkpoint",
        help="Direct-phase adapter_best.pt; required only for the dense phase.",
    )
    known, forwarded = parser.parse_known_args()
    config = load_yaml_mapping(known.config)
    if known.phase == "direct":
        if known.spatial_init_checkpoint:
            raise ValueError("--spatial-init-checkpoint is only valid for the dense phase.")
        _run_direct(config, forwarded)
        return
    if not known.spatial_init_checkpoint:
        raise ValueError("dense phase requires --spatial-init-checkpoint.")
    _run_dense(config, known.spatial_init_checkpoint, forwarded)


if __name__ == "__main__":
    main()
