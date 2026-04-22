from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.metrics import compute_training_reconstruction_metrics


class TestTrainingMetrics(unittest.TestCase):
    def test_reports_normalized_and_physical_metrics_separately(self) -> None:
        target = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=torch.float32)
        prediction = target + 0.1
        physical_target = target * 2.0 + 10.0
        normalization_cfg = {"mode": "zscore", "scope": "channel", "clip_min": None, "clip_max": None}

        metrics = compute_training_reconstruction_metrics(prediction, target, physical_target, normalization_cfg)

        self.assertIn("relative_l1", metrics)
        self.assertIn("normalized_relative_l1", metrics)
        self.assertIn("physical_relative_l1", metrics)
        self.assertAlmostEqual(metrics["relative_l1"], metrics["normalized_relative_l1"], places=6)
        self.assertNotAlmostEqual(metrics["normalized_relative_l1"], metrics["physical_relative_l1"], places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
