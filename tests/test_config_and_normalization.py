from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.data.normalization import denormalize_tensor, normalize_tensor


def load_release_config(name: str) -> dict:
    path = PROJECT_ROOT / "configs" / name
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def test_stage1_release_recipe_keeps_encoder_trainable() -> None:
    config = load_release_config("field_to_llm_stage1.yaml")
    alignment = config["patch_alignment"]

    assert alignment["fields"] == ["Vx", "Vy", "density", "pressure"]
    assert alignment["train_records"] == 3_500_000
    assert alignment["batch_size"] == 64
    assert alignment["eval_batch_size"] == 64
    assert alignment["encoder_source"] == "patch_ae_config"
    assert alignment["train_patch_ae"] is True
    assert alignment["freeze_patch_ae_after_pretrain"] is False
    assert alignment["patch_ae_pretrain_epochs"] == 2
    assert alignment["text_decimal_places"] == 2
    assert alignment["query_tokens"] == 256


def test_release_configs_expose_only_the_paper_interfaces() -> None:
    direct = load_release_config("field_to_llm_direct_qa.yaml")
    cross = load_release_config("field_to_llm_cross_attention.yaml")
    benchmark = load_release_config("field_to_llm_benchmark.yaml")

    assert direct["adapter"]["architecture"] == "alignment_adapter"
    assert direct["adapter"]["question_conditioning"] is False
    assert direct["adapter"]["structured_query_conditioning"] is False
    assert direct["llm_training"]["latent_channel_policy"] == "all"
    assert cross["memory"]["freeze_spatial_backbone"] is True
    assert cross["memory"]["latent_channel_policy"] == "all"
    assert cross["cross_attention"]["layers_1based"] == [8, 20, 32]
    assert cross["runtime"]["final_eval_reserve_minutes"] == 70
    assert benchmark["benchmark"]["methods"] == ["serialized", "dense"]
    assert benchmark["benchmark"]["max_records"] is None


def test_channelwise_minmax_normalizes_each_channel_independently() -> None:
    config = {"mode": "minmax", "scope": "channel", "clip_min": None, "clip_max": None}
    tensor = torch.tensor(
        [
            [[0.0, 2.0], [4.0, 6.0]],
            [[10.0, 14.0], [18.0, 22.0]],
        ]
    )

    normalized, _ = normalize_tensor(tensor, config)

    expected = torch.tensor(
        [
            [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
            [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
        ]
    )
    assert torch.allclose(normalized, expected, atol=1.0e-6)


def test_channelwise_zscore_roundtrips_with_denormalize() -> None:
    config = {"mode": "zscore", "scope": "channel", "clip_min": None, "clip_max": None}
    tensor = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[100.0, 120.0], [140.0, 160.0]],
        ]
    )

    normalized, state = normalize_tensor(tensor, config)
    restored = denormalize_tensor(normalized, state)

    assert torch.allclose(restored, tensor, atol=1.0e-5)
