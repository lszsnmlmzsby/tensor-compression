from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.utils.visualization import ReconstructionVisualizer2D


def _config(display_channel) -> dict:
    return {
        "data": {"dimensions": 2},
        "visualization": {
            "enabled": True,
            "num_samples": 2,
            "every_n_epochs": 1,
            "field_cmap": "turbo",
            "error_cmap": "inferno",
            "robust_percentile": 1.0,
            "display_channel": display_channel,
            "add_colorbar": False,
            "save_dirname": "reconstructions",
        },
    }


class TestVisualization(unittest.TestCase):
    def test_display_channel_all_expands_to_all_input_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ReconstructionVisualizer2D(
                config=_config("all"),
                run_dir=Path(tmpdir),
            )
            indices = visualizer._resolve_channel_indices(4)

        self.assertEqual(indices, [0, 1, 2, 3])

    def test_display_channel_integer_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ReconstructionVisualizer2D(
                config=_config(2),
                run_dir=Path(tmpdir),
            )
            indices = visualizer._resolve_channel_indices(4)

        self.assertEqual(indices, [2])

    def test_render_with_all_channels_creates_expected_number_of_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ReconstructionVisualizer2D(
                config=_config("all"),
                run_dir=Path(tmpdir),
            )
            inputs = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
            reconstructions = inputs.clone()

            fig = visualizer.render(inputs=inputs, reconstructions=reconstructions)

        self.assertEqual(len(fig.axes), 18)


if __name__ == "__main__":
    unittest.main(verbosity=2)
