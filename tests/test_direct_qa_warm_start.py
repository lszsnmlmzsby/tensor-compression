from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.train_tensor_llm_adapter import adapter_from_checkpoint, parse_args
from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_paper_direct_qa_config_resolves_to_prefix_warm_start(tmp_path: Path) -> None:
    alignment_checkpoint = tmp_path / "alignment_best.pt"
    environment = {
        "FIELD_TO_LLM_ROOT": str(tmp_path),
        "FIELD_TO_LLM_ALIGNMENT_CHECKPOINT": str(alignment_checkpoint),
    }
    argv = [
        "train_tensor_llm_adapter.py",
        "--config",
        str(PROJECT_ROOT / "configs" / "field_to_llm_direct_qa.yaml"),
    ]

    with patch.dict(os.environ, environment, clear=True), patch.object(sys, "argv", argv):
        args = parse_args()

    assert args.adapter_architecture == "alignment_adapter"
    assert args.adapter_init_checkpoint == str(alignment_checkpoint)
    assert args.stage2_warm_start_checkpoint is None
    assert args.stage2b_resume_checkpoint is None
    assert args.question_conditioning is False
    assert args.structured_query_conditioning is False
    assert args.max_train_records == 138240
    assert args.epochs == 1
    assert args.batch_size == 3
    assert args.lr == 2.0e-5
    assert args.ce_loss_weight == 0.05
    assert args.choice_ce_loss_weight == 1.0
    assert args.ranking_loss_weight == 0.1
    assert args.ranking_loss_negative == "no_latent"
    assert args.matched_group_loss_weight == 0.0
    assert args.full_local_reader_training is False


def test_direct_qa_machine_paths_override_portable_root(tmp_path: Path) -> None:
    paths = {
        "FIELD_TO_LLM_ROOT": str(tmp_path / "portable-root"),
        "FIELD_TO_LLM_ALIGNMENT_CHECKPOINT": str(tmp_path / "alignment_best.pt"),
        "FIELD_TO_LLM_HF_HOME": str(tmp_path / "hf"),
        "FIELD_TO_LLM_RUNS_DIR": str(tmp_path / "runs"),
        "FIELD_TO_LLM_MODEL_DIR": str(tmp_path / "Qwen2.5-14B-Instruct"),
        "FIELD_TO_LLM_DIRECT_QA_DIR": str(tmp_path / "direct-qa"),
        "FIELD_TO_LLM_LATENT_DIR": str(tmp_path / "latents"),
    }
    argv = [
        "train_tensor_llm_adapter.py",
        "--config",
        str(PROJECT_ROOT / "configs" / "field_to_llm_direct_qa.yaml"),
    ]

    with patch.dict(os.environ, paths, clear=True), patch.object(sys, "argv", argv):
        args = parse_args()

    assert args.model_name_or_path == paths["FIELD_TO_LLM_MODEL_DIR"]
    assert args.qa_dir == paths["FIELD_TO_LLM_DIRECT_QA_DIR"]
    assert args.latent_dir == paths["FIELD_TO_LLM_LATENT_DIR"]
    assert args.cache_dir == paths["FIELD_TO_LLM_HF_HOME"]
    assert args.hf_home == paths["FIELD_TO_LLM_HF_HOME"]
    assert args.output_root == paths["FIELD_TO_LLM_RUNS_DIR"]


def test_direct_qa_checkpoint_strictly_rebuilds_the_spatial_adapter() -> None:
    torch.manual_seed(17)
    source = TensorPatchAlignmentAdapter(
        latent_channels=3,
        latent_grid=(2, 2),
        adapter_dim=16,
        projection_dim=24,
        dropout=0.0,
        adapter_type="spatial_transformer",
        query_tokens=4,
        adapter_layers=1,
        adapter_heads=4,
        soft_prompt_scale=0.05,
    ).eval()
    checkpoint = {
        "args": {
            "adapter_architecture": "alignment_adapter",
            "adapter_type": "spatial_transformer",
            "adapter_dim": 16,
            "adapter_layers": 1,
            "adapter_heads": 4,
            "dropout": 0.0,
            "soft_prompt_tokens": 4,
            "soft_prompt_scale": 0.05,
        },
        "adapter_state_dict": source.state_dict(),
    }

    restored = adapter_from_checkpoint(
        checkpoint,
        latent_shape=(3, 2, 2),
        llm_hidden_size=24,
    ).eval()
    latent = torch.randn(2, 3, 2, 2)

    assert isinstance(restored, TensorPatchAlignmentAdapter)
    assert all(parameter.requires_grad for parameter in restored.parameters())
    torch.testing.assert_close(
        restored.forward_soft_prompts(latent),
        source.forward_soft_prompts(latent),
        rtol=0.0,
        atol=0.0,
    )
