from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.engine.tensor_editor_trainer import TensorEditorTrainer
from tensor_compression.models.editors.conditional_tensor_editor_2d import ConditionalTensorEditor2D


class DummyCompressor2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dim = 4

    def encode(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        latent_map = F.avg_pool2d(inputs, kernel_size=2).repeat(1, self.latent_dim, 1, 1)
        return {
            "latent_map": latent_map,
            "latent_tokens": latent_map.flatten(2).transpose(1, 2),
        }

    def decode(self, latent: dict[str, torch.Tensor]) -> torch.Tensor:
        return F.interpolate(latent["latent_map"][:, :1], scale_factor=2, mode="nearest")


def _editor_config() -> dict:
    return {
        "editor": {
            "compressor": {"freeze": True},
            "text": {
                "vocab_size": 128,
                "embed_dim": 8,
                "hidden_dim": 8,
                "max_length": 16,
                "dropout": 0.0,
            },
            "model": {
                "latent_grid": [2, 2],
                "latent_dim": 4,
                "latent_hidden_dim": 4,
                "num_res_blocks": 1,
                "condition_dim": 8,
                "activation": "gelu",
                "dropout": 0.0,
                "use_base_reconstruction": True,
                "residual_latent": True,
                "latent_delta_scale": 1.0,
                "zero_init_delta": True,
                "detach_latent_target": True,
            },
        }
    }


class TestLatentTensorEditor(unittest.TestCase):
    def test_zero_initialized_editor_starts_from_noisy_latent(self) -> None:
        model = ConditionalTensorEditor2D(compressor=DummyCompressor2D(), config=_editor_config())
        inputs = torch.randn(2, 1, 4, 4)

        outputs = model(inputs, ["add 0.2", "remove bias"])

        self.assertEqual(outputs["reconstruction"].shape, inputs.shape)
        self.assertEqual(outputs["latent_delta"].shape, outputs["latent_map"].shape)
        self.assertTrue(torch.allclose(outputs["latent_delta"], torch.zeros_like(outputs["latent_delta"])))
        self.assertTrue(torch.allclose(outputs["edited_latent_map"], outputs["latent_map"]))

    def test_latent_target_is_detached_when_compressor_is_frozen(self) -> None:
        model = ConditionalTensorEditor2D(compressor=DummyCompressor2D(), config=_editor_config())
        targets = torch.randn(2, 1, 4, 4, requires_grad=True)

        target_latent = model.encode_target(targets)

        self.assertFalse(target_latent["latent_map"].requires_grad)

    def test_trainer_adds_weighted_latent_mse_to_total_loss(self) -> None:
        trainer = object.__new__(TensorEditorTrainer)
        trainer.config = {"loss": {"weights": {"latent_mse": 0.5}}}
        outputs = {"edited_latent_map": torch.zeros(1, 2, 2, 2)}
        targets = torch.empty(1, 1, 4, 4)
        loss_dict = {"total": torch.tensor(1.0)}

        class TargetEncoder:
            def encode_target(self, _targets: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"latent_map": torch.ones_like(outputs["edited_latent_map"])}

        merged = TensorEditorTrainer._add_latent_loss(
            trainer,
            TargetEncoder(),
            outputs,
            targets,
            loss_dict,
        )

        self.assertAlmostEqual(float(merged["latent_mse"].item()), 1.0, places=6)
        self.assertAlmostEqual(float(merged["total"].item()), 1.5, places=6)

    def test_step_metrics_compare_input_base_and_editor(self) -> None:
        trainer = object.__new__(TensorEditorTrainer)
        trainer.config = {"loss": {"eps": 1.0e-6}}
        target = torch.zeros(1, 1, 2, 2)
        inputs = torch.ones_like(target) * 2.0
        outputs = {
            "reconstruction": torch.ones_like(target) * 0.5,
            "base_reconstruction": torch.ones_like(target),
            "latent_map": torch.ones(1, 2, 1, 1),
            "edited_latent_map": torch.ones(1, 2, 1, 1) * 1.5,
            "latent_delta": torch.ones(1, 2, 1, 1) * 0.5,
        }

        metrics = TensorEditorTrainer._build_step_metrics(
            trainer,
            loss_dict=None,
            outputs=outputs,
            inputs=inputs,
            target=target,
        )

        self.assertAlmostEqual(metrics["input_mae"], 2.0, places=6)
        self.assertAlmostEqual(metrics["base_mae"], 1.0, places=6)
        self.assertAlmostEqual(metrics["editor_mae"], 0.5, places=6)
        self.assertAlmostEqual(metrics["gain_vs_input_mae"], 0.75, places=6)
        self.assertAlmostEqual(metrics["gain_vs_base_mae"], 0.5, places=6)
        self.assertAlmostEqual(metrics["mae_reduction_vs_input"], 1.5, places=6)
        self.assertAlmostEqual(metrics["mae_reduction_vs_base"], 0.5, places=6)
        self.assertAlmostEqual(metrics["latent_delta_rms"], 0.5, places=6)

    def test_groups_validation_indices_by_meta_type(self) -> None:
        trainer = object.__new__(TensorEditorTrainer)
        grouped = TensorEditorTrainer._group_batch_indices_by_type(
            trainer,
            [{"type": "normal"}, {"type": "add_offset"}, {"perturbation_type": "add_offset"}, {}],
        )

        self.assertEqual(grouped["normal"], [0])
        self.assertEqual(grouped["add_offset"], [1, 2])
        self.assertEqual(grouped["unknown"], [3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
