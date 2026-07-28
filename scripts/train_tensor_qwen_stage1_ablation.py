from __future__ import annotations

"""Stage-1 alignment ablation without changing either production trainer.

``direct`` replaces only the in-memory Stage-1 spatial-adapter state with a
deterministically initialized, architecture-identical state.  The original
checkpoint is still used for latent/encoder provenance validation.  ``dense``
then runs the production dense cross-attention trainer from that direct-QA
checkpoint and refuses checkpoints which do not carry this ablation contract.
"""

import argparse
import copy
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402
from tensor_compression.downstream.patch_qa_contract import sha256_file  # noqa: E402
from tensor_compression.utils.pipeline_config import load_yaml_mapping, resolve_path_string  # noqa: E402


ABLATION_MODE = "random_spatial_adapter_preserve_latent_encoder"


def _resolved(value: Any) -> Path:
    return Path(resolve_path_string(value, PROJECT_ROOT)).expanduser().resolve()


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _random_spatial_state(payload: Mapping[str, Any], seed: int) -> dict[str, torch.Tensor]:
    state = payload.get("adapter_state_dict")
    checkpoint_args = payload.get("args")
    if not isinstance(state, Mapping) or not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-1 checkpoint is missing args or adapter_state_dict.")
    latent_weight = state.get("latent_projection.weight")
    output_weight = state.get("output.1.weight")
    position = state.get("spatial_pos_encoding")
    if not all(isinstance(value, torch.Tensor) for value in (latent_weight, output_weight, position)):
        raise ValueError("Stage-1 ablation requires a spatial_transformer alignment checkpoint.")
    token_count = int(position.shape[-2])
    side = int(round(token_count**0.5))
    if side * side != token_count:
        raise ValueError(f"Spatial token count must form a square grid, got {token_count}.")
    adapter_dim = int(latent_weight.shape[0])
    heads = int(checkpoint_args.get("adapter_heads", 8))
    layers = max(
        [int(str(key).split(".")[1]) + 1 for key in state if str(key).startswith("blocks.")]
        or [int(checkpoint_args.get("adapter_layers", 2))]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=int(latent_weight.shape[1]),
            latent_grid=(side, side),
            adapter_dim=adapter_dim,
            projection_dim=int(output_weight.shape[0]),
            dropout=float(checkpoint_args.get("dropout", 0.0)),
            adapter_type="spatial_transformer",
            query_tokens=token_count,
            adapter_layers=layers,
            adapter_heads=heads,
            soft_prompt_scale=float(checkpoint_args.get("soft_prompt_scale", 0.05)),
        )
    fresh = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
    if set(fresh) != set(state):
        raise ValueError(
            "Fresh adapter does not exactly match the Stage-1 architecture; "
            f"missing={sorted(set(state) - set(fresh))}, extra={sorted(set(fresh) - set(state))}."
        )
    return fresh


def _ablation_contract(source: Path, seed: int, state_sha256: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "factor_removed": "stage1_tensor_text_alignment",
        "mode": ABLATION_MODE,
        "seed": int(seed),
        "source_stage1_checkpoint": str(source),
        "source_stage1_sha256": sha256_file(source),
        "random_adapter_state_sha256": state_sha256,
        "preserved": [
            "cached_latents",
            "frozen_field_encoder",
            "spatial_adapter_architecture",
            "direct_qa_training",
        ],
    }


def _run_direct(config: Mapping[str, Any], forwarded: list[str]) -> None:
    import scripts.train_tensor_llm_adapter as trainer

    section = config.get("ablation", {})
    direct = config.get("direct", {})
    if not isinstance(section, Mapping) or not isinstance(direct, Mapping):
        raise ValueError("Ablation config requires mapping sections: ablation and direct.")
    source = _resolved(section.get("source_stage1_checkpoint"))
    seed = int(section.get("seed", 42))
    base_config = str(_resolved(direct.get("base_config")))
    overrides = direct.get("arg_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("direct.arg_overrides must be a mapping of resolved argument names.")

    original_parse = trainer.parse_args
    original_load = torch.load
    original_save = trainer.atomic_torch_save
    cache: dict[str, Any] = {}

    def parse_args() -> argparse.Namespace:
        args = original_parse()
        forwarded_options = {
            token.split("=", 1)[0]
            for token in forwarded
            if token.startswith("--")
        }
        for name, value in overrides.items():
            if not hasattr(args, str(name)):
                raise ValueError(f"Unknown direct trainer argument override: {name}")
            option = "--" + str(name).replace("_", "-")
            if option in forwarded_options or "--no-" + option[2:] in forwarded_options:
                continue
            setattr(args, str(name), copy.deepcopy(value))
        configured_source = Path(str(args.adapter_init_checkpoint)).expanduser().resolve()
        if configured_source != source:
            raise ValueError(
                "The direct base config adapter.init_checkpoint must equal the ablation "
                f"source checkpoint: configured={configured_source}, source={source}."
            )
        args.stage1_ablation = True
        args.stage1_ablation_mode = ABLATION_MODE
        args.stage1_ablation_seed = seed
        args.stage1_ablation_source_checkpoint = str(source)
        return args

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        payload = original_load(path, *args, **kwargs)
        try:
            matches = Path(path).expanduser().resolve() == source
        except (TypeError, OSError):
            matches = False
        if not matches:
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("Stage-1 checkpoint payload must be a mapping.")
        if "state" not in cache:
            cache["state"] = _random_spatial_state(payload, seed)
            cache["contract"] = _ablation_contract(source, seed, _state_digest(cache["state"]))
        replaced = dict(payload)
        replaced["adapter_state_dict"] = cache["state"]
        replaced["ablation_contract"] = cache["contract"]
        return replaced

    def save(path: Any, payload: Mapping[str, Any]) -> None:
        enriched = dict(payload)
        if enriched.get("checkpoint_type") == "tensor_llm_adapter":
            if "contract" not in cache:
                raise RuntimeError("Ablation adapter was never loaded before checkpoint save.")
            enriched["ablation_contract"] = copy.deepcopy(cache["contract"])
        original_save(path, enriched)

    trainer.parse_args = parse_args
    trainer.atomic_torch_save = save
    torch.load = load
    sys.argv = [sys.argv[0], "--config", base_config, *forwarded]
    trainer.main()


def _run_dense(config: Mapping[str, Any], init_checkpoint: str, forwarded: list[str]) -> None:
    import scripts.train_tensor_qwen_cross_attention as trainer

    dense = config.get("dense", {})
    if not isinstance(dense, Mapping):
        raise ValueError("Ablation config requires a dense mapping section.")
    base_config = str(_resolved(dense.get("base_config")))
    init_path = Path(init_checkpoint).expanduser().resolve()
    payload = torch.load(init_path, map_location="cpu", weights_only=True)
    contract = payload.get("ablation_contract") if isinstance(payload, Mapping) else None
    if not isinstance(contract, Mapping) or contract.get("mode") != ABLATION_MODE:
        raise ValueError(
            "Dense Stage-1 ablation requires adapter_best.pt produced by this script's direct phase."
        )

    original_parse = trainer.parse_args

    def parse_args() -> argparse.Namespace:
        args = original_parse()
        args.memory_init_checkpoint = str(init_path)
        args.stage1_ablation = True
        args.stage1_ablation_contract = copy.deepcopy(dict(contract))
        raw = copy.deepcopy(args.raw_config)
        raw["stage1_ablation"] = copy.deepcopy(dict(contract))
        args.raw_config = raw
        trainer.validate_args(args)
        return args

    trainer.parse_args = parse_args
    sys.argv = [sys.argv[0], "--config", base_config, *forwarded]
    trainer.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("direct", "dense"))
    parser.add_argument("--config", required=True, help="Stage-1 ablation orchestration YAML.")
    parser.add_argument(
        "--spatial-init-checkpoint",
        help="Direct-phase adapter_best.pt; required for the dense phase.",
    )
    known, forwarded = parser.parse_known_args()
    config = load_yaml_mapping(known.config)
    if known.phase == "direct":
        if known.spatial_init_checkpoint:
            raise ValueError("--spatial-init-checkpoint is only valid for the dense phase.")
        _run_direct(config, forwarded)
    else:
        if not known.spatial_init_checkpoint:
            raise ValueError("dense phase requires --spatial-init-checkpoint.")
        _run_dense(config, known.spatial_init_checkpoint, forwarded)


if __name__ == "__main__":
    main()
