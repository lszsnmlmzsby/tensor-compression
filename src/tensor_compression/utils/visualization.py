from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class ReconstructionVisualizer2D:
    def __init__(self, config: dict, run_dir: Path) -> None:
        vis_cfg = config["visualization"]
        self.enabled = bool(vis_cfg["enabled"])
        self.num_samples = int(vis_cfg["num_samples"])
        self.every_n_epochs = int(vis_cfg["every_n_epochs"])
        self.field_cmap = str(vis_cfg["field_cmap"])
        self.error_cmap = str(vis_cfg["error_cmap"])
        self.robust_percentile = float(vis_cfg["robust_percentile"])
        self.display_channel = int(vis_cfg["display_channel"])
        self.add_colorbar = bool(vis_cfg["add_colorbar"])
        self.output_dir = run_dir / vis_cfg["save_dirname"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_run(self, epoch: int) -> bool:
        return self.enabled and (epoch % self.every_n_epochs == 0)

    def save(self, inputs: torch.Tensor, reconstructions: torch.Tensor, epoch: int) -> Path:
        fig = self.render(inputs=inputs, reconstructions=reconstructions)
        path = self.save_figure(fig=fig, epoch=epoch)
        plt.close(fig)
        return path

    def render(self, inputs: torch.Tensor, reconstructions: torch.Tensor):
        inputs = inputs.detach().cpu()
        reconstructions = reconstructions.detach().cpu()
        count = min(self.num_samples, inputs.shape[0])
        fig, axes = plt.subplots(count, 3, figsize=(12, 4 * count), constrained_layout=True)
        if count == 1:
            axes = [axes]
        for row in range(count):
            inp = self._to_scalar_field(inputs[row])
            rec = self._to_scalar_field(reconstructions[row])
            err = np.abs(rec - inp)
            field_vmin, field_vmax = self._robust_range(np.concatenate([inp.ravel(), rec.ravel()]))
            err_vmin, err_vmax = 0.0, self._robust_upper(err)

            im0 = axes[row][0].imshow(inp, cmap=self.field_cmap, vmin=field_vmin, vmax=field_vmax)
            im1 = axes[row][1].imshow(rec, cmap=self.field_cmap, vmin=field_vmin, vmax=field_vmax)
            im2 = axes[row][2].imshow(err, cmap=self.error_cmap, vmin=err_vmin, vmax=err_vmax)

            axes[row][0].set_title("original field")
            axes[row][1].set_title("reconstructed field")
            axes[row][2].set_title("absolute difference")

            if self.add_colorbar:
                fig.colorbar(im0, ax=axes[row][0], fraction=0.046, pad=0.04)
                fig.colorbar(im1, ax=axes[row][1], fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=axes[row][2], fraction=0.046, pad=0.04)
            for col in range(3):
                axes[row][col].axis("off")
        return fig

    def save_figure(self, fig, epoch: int) -> Path:
        path = self.output_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def _to_scalar_field(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim != 3:
            raise ValueError(f"Expected CHW tensor for visualization, got {tuple(tensor.shape)}.")
        channel_index = min(self.display_channel, tensor.shape[0] - 1)
        return tensor[channel_index].numpy()

    def _robust_range(self, values: np.ndarray) -> tuple[float, float]:
        low = np.percentile(values, self.robust_percentile)
        high = np.percentile(values, 100.0 - self.robust_percentile)
        if np.isclose(low, high):
            low = float(values.min())
            high = float(values.max())
        if np.isclose(low, high):
            high = low + 1.0e-6
        return float(low), float(high)

    def _robust_upper(self, values: np.ndarray) -> float:
        upper = np.percentile(values, 100.0 - self.robust_percentile)
        if np.isclose(upper, 0.0):
            upper = float(values.max())
        if np.isclose(upper, 0.0):
            upper = 1.0e-6
        return float(upper)


class ReconstructionVisualizer3D:
    AXIS_NAMES = ("axial", "coronal", "sagittal")

    def __init__(self, config: dict, run_dir: Path) -> None:
        vis_cfg = config["visualization"]
        self.enabled = bool(vis_cfg["enabled"])
        self.num_samples = int(vis_cfg["num_samples"])
        self.every_n_epochs = int(vis_cfg["every_n_epochs"])
        self.field_cmap = str(vis_cfg["field_cmap"])
        self.error_cmap = str(vis_cfg["error_cmap"])
        self.robust_percentile = float(vis_cfg["robust_percentile"])
        self.display_channel = int(vis_cfg["display_channel"])
        self.add_colorbar = bool(vis_cfg["add_colorbar"])
        self.output_dir = run_dir / vis_cfg["save_dirname"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_run(self, epoch: int) -> bool:
        return self.enabled and (epoch % self.every_n_epochs == 0)

    def save(self, inputs: torch.Tensor, reconstructions: torch.Tensor, epoch: int) -> Path:
        fig = self.render(inputs=inputs, reconstructions=reconstructions)
        path = self.save_figure(fig=fig, epoch=epoch)
        plt.close(fig)
        return path

    def render(self, inputs: torch.Tensor, reconstructions: torch.Tensor):
        inputs = inputs.detach().cpu()
        reconstructions = reconstructions.detach().cpu()
        count = min(self.num_samples, inputs.shape[0])

        fig, axes = plt.subplots(
            count * 3,
            3,
            figsize=(12, 4 * count * 3),
            constrained_layout=True,
        )
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(1, 3)
        axes = axes.reshape(count, 3, 3)

        for sample_idx in range(count):
            inp = self._to_scalar_volume(inputs[sample_idx])
            rec = self._to_scalar_volume(reconstructions[sample_idx])
            err = np.abs(rec - inp)

            field_vmin, field_vmax = self._robust_range(
                np.concatenate([inp.ravel(), rec.ravel()])
            )
            err_vmin, err_vmax = 0.0, self._robust_upper(err)

            for view_idx, axis_name in enumerate(self.AXIS_NAMES):
                inp_slice = self._extract_mid_slice(inp, view_idx)
                rec_slice = self._extract_mid_slice(rec, view_idx)
                err_slice = self._extract_mid_slice(err, view_idx)

                row_axes = axes[sample_idx, view_idx]
                im0 = row_axes[0].imshow(
                    inp_slice,
                    cmap=self.field_cmap,
                    vmin=field_vmin,
                    vmax=field_vmax,
                    origin="lower",
                )
                im1 = row_axes[1].imshow(
                    rec_slice,
                    cmap=self.field_cmap,
                    vmin=field_vmin,
                    vmax=field_vmax,
                    origin="lower",
                )
                im2 = row_axes[2].imshow(
                    err_slice,
                    cmap=self.error_cmap,
                    vmin=err_vmin,
                    vmax=err_vmax,
                    origin="lower",
                )

                titles = (
                    f"{axis_name}: original",
                    f"{axis_name}: reconstruction",
                    f"{axis_name}: abs diff",
                )
                for col_idx, title in enumerate(titles):
                    row_axes[col_idx].set_title(title)
                    row_axes[col_idx].axis("off")

                if self.add_colorbar:
                    fig.colorbar(im0, ax=row_axes[0], fraction=0.046, pad=0.04)
                    fig.colorbar(im1, ax=row_axes[1], fraction=0.046, pad=0.04)
                    fig.colorbar(im2, ax=row_axes[2], fraction=0.046, pad=0.04)

        return fig

    def save_figure(self, fig, epoch: int) -> Path:
        path = self.output_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def _to_scalar_volume(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim != 4:
            raise ValueError(f"Expected CDHW tensor for visualization, got {tuple(tensor.shape)}.")
        channel_index = min(self.display_channel, tensor.shape[0] - 1)
        return tensor[channel_index].numpy()

    def _extract_mid_slice(self, volume: np.ndarray, axis: int) -> np.ndarray:
        mid_index = volume.shape[axis] // 2
        if axis == 0:
            slice_2d = volume[mid_index, :, :]
        elif axis == 1:
            slice_2d = volume[:, mid_index, :]
        elif axis == 2:
            slice_2d = volume[:, :, mid_index]
        else:
            raise ValueError(f"Unsupported axis for 3D visualization: {axis}")
        return np.asarray(slice_2d)

    def _robust_range(self, values: np.ndarray) -> tuple[float, float]:
        low = np.percentile(values, self.robust_percentile)
        high = np.percentile(values, 100.0 - self.robust_percentile)
        if np.isclose(low, high):
            low = float(values.min())
            high = float(values.max())
        if np.isclose(low, high):
            high = low + 1.0e-6
        return float(low), float(high)

    def _robust_upper(self, values: np.ndarray) -> float:
        upper = np.percentile(values, 100.0 - self.robust_percentile)
        if np.isclose(upper, 0.0):
            upper = float(values.max())
        if np.isclose(upper, 0.0):
            upper = 1.0e-6
        return float(upper)


class _UnsupportedVisualizer:
    def should_run(self, epoch: int) -> bool:
        return False

    def save(self, inputs: torch.Tensor, reconstructions: torch.Tensor, epoch: int):
        raise NotImplementedError("Visualizer is not implemented for this tensor dimensionality yet.")

    def render(self, inputs: torch.Tensor, reconstructions: torch.Tensor):
        raise NotImplementedError("Visualizer is not implemented for this tensor dimensionality yet.")

    def save_figure(self, fig, epoch: int):
        raise NotImplementedError("Visualizer is not implemented for this tensor dimensionality yet.")


def build_visualizer(config: dict, run_dir: Path):
    dimensions = int(config["data"]["dimensions"])
    if dimensions == 2:
        return ReconstructionVisualizer2D(config=config, run_dir=run_dir)
    if dimensions == 3:
        return ReconstructionVisualizer3D(config=config, run_dir=run_dir)
    return _UnsupportedVisualizer()
