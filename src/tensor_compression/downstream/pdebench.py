from __future__ import annotations

import importlib
import importlib.util
import math
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from tensor_compression.config import load_config
from tensor_compression.data.normalization import denormalize_tensor, normalize_tensor
from tensor_compression.metrics import compute_reconstruction_metrics
from tensor_compression.models import build_model

COORDINATE_KEY_SUFFIX = "-coordinate"
DEFAULT_COMPRESSIBLE_FIELDS = ("density", "pressure", "Vx", "Vy", "Vz")


@dataclass(frozen=True)
class PDEBenchField:
    path: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class PDEBenchRecord:
    sample_index: int
    original: torch.Tensor
    reconstructed: torch.Tensor
    grid: torch.Tensor | None
    t_coordinates: torch.Tensor | None
    field_names: tuple[str, ...]
    layout: str = "pdebench_spatial_time_channel"

    def as_payload(self, variant: str) -> dict[str, Any]:
        if variant not in {"original", "reconstructed"}:
            raise ValueError(f"Unsupported payload variant: {variant}")
        data = self.original if variant == "original" else self.reconstructed
        return {
            "data": data,
            "grid": self.grid,
            "t_coordinates": self.t_coordinates,
            "field_names": self.field_names,
            "sample_index": self.sample_index,
            "layout": self.layout,
        }


class CheckpointReconstructor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        project_root: str | Path | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Compressor checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_path}")

        raw_config = checkpoint.get("config")
        if config_path is not None:
            root = Path(project_root).resolve() if project_root is not None else None
            self.config = load_config(config_path, base_root=root)
        elif isinstance(raw_config, Mapping):
            self.config = dict(raw_config)
        else:
            raise ValueError(
                "Compressor checkpoint does not include a `config`. "
                "Pass --compressor-config explicitly."
            )

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError(f"Checkpoint does not contain `model_state_dict`: {self.checkpoint_path}")

        self.device = resolve_device(device)
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.input_size = tuple(int(dim) for dim in self.config["model"]["input_size"])
        self.channels = int(self.config["model"]["in_channels"])
        self.normalization_cfg = dict(self.config.get("data", {}).get("dataset", {}).get("normalization", {}))

    @torch.no_grad()
    def reconstruct_frames(self, frames: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"Expected frames shaped [frames, channels, H, W], got {tuple(frames.shape)}")
        if frames.shape[1] != self.channels:
            raise ValueError(
                f"Compressor expects {self.channels} channels but got {frames.shape[1]}."
            )

        original_hw = tuple(int(dim) for dim in frames.shape[-2:])
        frames = frames.to(self.device, dtype=torch.float32)
        resized = resize_chw_batch(frames, self.input_size)
        normalized_frames: list[torch.Tensor] = []
        normalization_states: list[dict[str, torch.Tensor | float | str | None]] = []
        for frame in resized:
            normalized_frame, state = normalize_tensor(frame.cpu(), self.normalization_cfg)
            normalized_frames.append(normalized_frame)
            normalization_states.append(state)
        normalized = torch.stack([frame.to(self.device) for frame in normalized_frames], dim=0)
        outputs: list[torch.Tensor] = []
        for start in range(0, normalized.shape[0], max(1, batch_size)):
            batch = normalized[start : start + max(1, batch_size)]
            reconstruction = self.model(batch)["reconstruction"]
            batch_states = normalization_states[start : start + max(1, batch_size)]
            reconstruction = torch.stack(
                [
                    denormalize_tensor(sample, state)
                    for sample, state in zip(reconstruction, batch_states)
                ],
                dim=0,
            )
            if tuple(reconstruction.shape[-2:]) != original_hw:
                reconstruction = F.interpolate(
                    reconstruction,
                    size=original_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(reconstruction.detach().cpu())
        return torch.cat(outputs, dim=0)


class ExternalCallableOperator:
    def __init__(self, spec: str, device: str | torch.device = "auto") -> None:
        self.spec = spec
        self.device = resolve_device(device)
        self.callable = load_callable_from_spec(spec)
        if hasattr(self.callable, "to"):
            self.callable = self.callable.to(self.device)
        if hasattr(self.callable, "eval"):
            self.callable.eval()

    @torch.no_grad()
    def __call__(self, payload: dict[str, Any]) -> Any:
        return self.callable(payload)


class PDEBenchFNOForwardOperator:
    def __init__(
        self,
        pdebench_root: str | Path,
        checkpoint_path: str | Path,
        num_channels: int,
        initial_step: int = 10,
        modes: int = 12,
        width: int = 20,
        t_train: int | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        add_pdebench_to_syspath(pdebench_root)
        from pdebench.models.fno.fno import FNO1d, FNO2d, FNO3d

        self.device = resolve_device(device)
        self.initial_step = int(initial_step)
        self.t_train = t_train
        self.num_channels = int(num_channels)
        self.modes = int(modes)
        self.width = int(width)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PDEBench FNO checkpoint not found: {self.checkpoint_path}")
        self.model_classes = {1: FNO1d, 2: FNO2d, 3: FNO3d}
        self.model: torch.nn.Module | None = None
        self.spatial_dim: int | None = None

    @torch.no_grad()
    def __call__(self, payload: dict[str, Any]) -> torch.Tensor:
        data = ensure_batched_pdebench_tensor(
            payload["data"],
            already_batched=bool(payload.get("batched", False)),
        ).to(self.device)
        grid = payload.get("grid")
        if grid is None:
            grid = make_unit_grid(data.shape[1:-2], self.device)
        else:
            grid = grid.to(self.device, dtype=torch.float32)
        if grid.ndim == data.ndim - 2:
            grid = grid.unsqueeze(0).expand(data.shape[0], *grid.shape)

        spatial_dim = data.ndim - 3
        model = self._ensure_model(spatial_dim)
        t_limit = min(int(self.t_train or data.shape[-2]), data.shape[-2])
        if t_limit <= self.initial_step:
            raise ValueError(
                f"t_train={t_limit} must be greater than initial_step={self.initial_step}."
            )

        history = data[..., : self.initial_step, :]
        prediction = data[..., : self.initial_step, :]
        input_shape = list(history.shape[:-2]) + [-1]
        for _ in range(self.initial_step, t_limit):
            inp = history.reshape(input_shape)
            next_frame = model(inp, grid)
            prediction = torch.cat((prediction, next_frame), dim=-2)
            history = torch.cat((history[..., 1:, :], next_frame), dim=-2)
        return prediction.detach().cpu()

    def _ensure_model(self, spatial_dim: int) -> torch.nn.Module:
        if self.model is not None and self.spatial_dim == spatial_dim:
            return self.model
        if spatial_dim not in self.model_classes:
            raise ValueError(f"Unsupported FNO spatial dimension: {spatial_dim}")
        cls = self.model_classes[spatial_dim]
        if spatial_dim == 1:
            model = cls(
                num_channels=self.num_channels,
                modes=self.modes,
                width=self.width,
                initial_step=self.initial_step,
            )
        elif spatial_dim == 2:
            model = cls(
                num_channels=self.num_channels,
                modes1=self.modes,
                modes2=self.modes,
                width=self.width,
                initial_step=self.initial_step,
            )
        else:
            model = cls(
                num_channels=self.num_channels,
                modes1=self.modes,
                modes2=self.modes,
                modes3=self.modes,
                width=self.width,
                initial_step=self.initial_step,
            )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        self.spatial_dim = spatial_dim
        return model


class PDEBenchUNetForwardOperator:
    def __init__(
        self,
        pdebench_root: str | Path,
        checkpoint_path: str | Path,
        num_channels: int,
        initial_step: int = 10,
        t_train: int | None = None,
        init_features: int = 32,
        device: str | torch.device = "auto",
    ) -> None:
        add_pdebench_to_syspath(pdebench_root)
        from pdebench.models.unet.unet import UNet1d, UNet2d, UNet3d

        self.device = resolve_device(device)
        self.initial_step = int(initial_step)
        self.t_train = t_train
        self.num_channels = int(num_channels)
        self.init_features = int(init_features)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PDEBench UNet checkpoint not found: {self.checkpoint_path}")
        self.model_classes = {1: UNet1d, 2: UNet2d, 3: UNet3d}
        self.model: torch.nn.Module | None = None
        self.spatial_dim: int | None = None

    @torch.no_grad()
    def __call__(self, payload: dict[str, Any]) -> torch.Tensor:
        data = ensure_batched_pdebench_tensor(
            payload["data"],
            already_batched=bool(payload.get("batched", False)),
        ).to(self.device)
        spatial_dim = data.ndim - 3
        model = self._ensure_model(spatial_dim)
        t_limit = min(int(self.t_train or data.shape[-2]), data.shape[-2])
        if t_limit <= self.initial_step:
            raise ValueError(
                f"t_train={t_limit} must be greater than initial_step={self.initial_step}."
            )

        history = data[..., : self.initial_step, :]
        prediction = data[..., : self.initial_step, :]
        input_shape = list(history.shape[:-2]) + [-1]
        for _ in range(self.initial_step, t_limit):
            inp = history.reshape(input_shape)
            permutation = [0, -1, *range(1, inp.ndim - 1)]
            inp = inp.permute(permutation)
            inverse_permutation = [0, *range(2, inp.ndim), 1]
            next_frame = model(inp).permute(inverse_permutation).unsqueeze(-2)
            prediction = torch.cat((prediction, next_frame), dim=-2)
            history = torch.cat((history[..., 1:, :], next_frame), dim=-2)
        return prediction.detach().cpu()

    def _ensure_model(self, spatial_dim: int) -> torch.nn.Module:
        if self.model is not None and self.spatial_dim == spatial_dim:
            return self.model
        if spatial_dim not in self.model_classes:
            raise ValueError(f"Unsupported UNet spatial dimension: {spatial_dim}")
        cls = self.model_classes[spatial_dim]
        model = cls(
            in_channels=self.num_channels * self.initial_step,
            out_channels=self.num_channels,
            init_features=self.init_features,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model
        self.spatial_dim = spatial_dim
        return model


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    requested = str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def inspect_pdebench_fields(
    hdf5_path: str | Path,
    field_keys: Sequence[str] | None = None,
    include_scalar_2d: bool = True,
) -> list[PDEBenchField]:
    path = Path(hdf5_path).expanduser()
    fields: list[PDEBenchField] = []
    with h5py.File(path, "r") as handle:
        selected = list(field_keys) if field_keys else discover_compressible_field_keys(handle)
        for key in selected:
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise KeyError(f"HDF5 dataset key {key!r} not found in {path}.")
            dataset = handle[key]
            if not np.issubdtype(dataset.dtype, np.number):
                continue
            if dataset.ndim < 2:
                continue
            if not include_scalar_2d and dataset.ndim == 2:
                continue
            fields.append(
                PDEBenchField(
                    path=key,
                    shape=tuple(int(dim) for dim in dataset.shape),
                    dtype=str(dataset.dtype),
                )
            )
    return fields


def discover_compressible_field_keys(handle: h5py.File) -> list[str]:
    top_level_keys = list(handle.keys())
    preferred = [
        key for key in DEFAULT_COMPRESSIBLE_FIELDS if key in handle and isinstance(handle[key], h5py.Dataset)
    ]
    if preferred:
        return preferred

    fields: list[str] = []

    def visitor(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if name.endswith(COORDINATE_KEY_SUFFIX) or name in {"t-coordinate"}:
            return
        if not np.issubdtype(obj.dtype, np.number):
            return
        if obj.ndim < 2:
            return
        fields.append(name)

    handle.visititems(visitor)
    fields.sort(key=lambda key: top_level_keys.index(key) if key in top_level_keys else len(top_level_keys))
    return fields


def read_pdebench_sample(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    sample_index: int,
    time_slice: slice | None = None,
    spatial_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    path = Path(hdf5_path).expanduser()
    with h5py.File(path, "r") as handle:
        arrays = []
        reference_shape: tuple[int, ...] | None = None
        for key in field_keys:
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise KeyError(f"HDF5 dataset key {key!r} not found in {path}.")
            dataset = handle[key]
            if dataset.ndim < 3:
                raise ValueError(
                    f"Expected field {key!r} to be at least [sample, time, ...], got {dataset.shape}."
                )
            if reference_shape is None:
                reference_shape = tuple(int(dim) for dim in dataset.shape)
            elif tuple(int(dim) for dim in dataset.shape) != reference_shape:
                raise ValueError("All selected PDEBench fields must have matching shapes.")
            indexer = build_sample_indexer(dataset.ndim, sample_index, time_slice, spatial_stride)
            arrays.append(np.asarray(dataset[tuple(indexer)], dtype=np.float32))
        stacked = np.stack(arrays, axis=-1)
        data = torch.as_tensor(np.moveaxis(stacked, 0, -2), dtype=torch.float32)
        grid = read_grid(handle, data.shape[:-2], spatial_stride)
        t_coordinates = read_time_coordinates(handle, time_slice)
    return data, grid, t_coordinates


def build_sample_indexer(
    ndim: int,
    sample_index: int,
    time_slice: slice | None,
    spatial_stride: int,
) -> list[int | slice]:
    if ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions, got ndim={ndim}")
    if spatial_stride <= 0:
        raise ValueError(f"spatial_stride must be positive, got {spatial_stride}")
    indexer: list[int | slice] = [int(sample_index), time_slice or slice(None)]
    indexer.extend(slice(None, None, spatial_stride) for _ in range(ndim - 2))
    return indexer


def read_grid(
    handle: h5py.File,
    spatial_shape: Sequence[int],
    spatial_stride: int = 1,
) -> torch.Tensor | None:
    coordinate_names = ["x-coordinate", "y-coordinate", "z-coordinate"][: len(spatial_shape)]
    if not all(name in handle for name in coordinate_names):
        return None
    coordinates = [
        torch.as_tensor(np.asarray(handle[name][::spatial_stride], dtype=np.float32))
        for name in coordinate_names
    ]
    if any(int(coord.numel()) != int(size) for coord, size in zip(coordinates, spatial_shape)):
        return None
    mesh = torch.meshgrid(*coordinates, indexing="ij")
    return torch.stack(mesh, dim=-1)


def read_time_coordinates(handle: h5py.File, time_slice: slice | None = None) -> torch.Tensor | None:
    if "t-coordinate" not in handle:
        return None
    values = np.asarray(handle["t-coordinate"][()], dtype=np.float32)
    if time_slice is not None:
        values = values[time_slice]
    return torch.as_tensor(values, dtype=torch.float32)


def build_pdebench_record(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    sample_index: int,
    reconstructor: CheckpointReconstructor | None = None,
    batch_size: int = 1,
    time_slice: slice | None = None,
    spatial_stride: int = 1,
) -> PDEBenchRecord:
    original, grid, t_coordinates = read_pdebench_sample(
        hdf5_path=hdf5_path,
        field_keys=field_keys,
        sample_index=sample_index,
        time_slice=time_slice,
        spatial_stride=spatial_stride,
    )
    reconstructed = original.clone()
    if reconstructor is not None:
        frames = pdebench_to_chw_frames(original)
        reconstructed_frames = reconstructor.reconstruct_frames(frames, batch_size=batch_size)
        reconstructed = chw_frames_to_pdebench(
            reconstructed_frames,
            spatial_shape=original.shape[:-2],
            time_steps=original.shape[-2],
        )
    return PDEBenchRecord(
        sample_index=sample_index,
        original=original,
        reconstructed=reconstructed,
        grid=grid,
        t_coordinates=t_coordinates,
        field_names=tuple(field_keys),
    )


def generate_pdebench_records(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    sample_indices: Iterable[int],
    reconstructor: CheckpointReconstructor | None = None,
    batch_size: int = 1,
    time_slice: slice | None = None,
    spatial_stride: int = 1,
):
    for sample_index in sample_indices:
        yield build_pdebench_record(
            hdf5_path=hdf5_path,
            field_keys=field_keys,
            sample_index=sample_index,
            reconstructor=reconstructor,
            batch_size=batch_size,
            time_slice=time_slice,
            spatial_stride=spatial_stride,
        )


def iter_pdebench_records(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    sample_indices: Sequence[int],
    reconstructor: CheckpointReconstructor | None = None,
    batch_size: int = 1,
    time_slice: slice | None = None,
    spatial_stride: int = 1,
) -> list[PDEBenchRecord]:
    return list(
        generate_pdebench_records(
            hdf5_path=hdf5_path,
            field_keys=field_keys,
            sample_indices=sample_indices,
            reconstructor=reconstructor,
            batch_size=batch_size,
            time_slice=time_slice,
            spatial_stride=spatial_stride,
        )
    )


def export_reconstructed_hdf5(
    hdf5_path: str | Path,
    output_path: str | Path,
    records: Sequence[PDEBenchRecord],
    field_keys: Sequence[str],
    time_slice: slice | None = None,
    spatial_stride: int = 1,
    overwrite: bool = False,
) -> Path:
    target_path = prepare_reconstructed_hdf5_output(
        hdf5_path=hdf5_path,
        output_path=output_path,
        overwrite=overwrite,
    )
    with h5py.File(target_path, "r+") as target:
        for record in records:
            write_reconstructed_record_to_hdf5(
                target=target,
                record=record,
                field_keys=field_keys,
                time_slice=time_slice,
                spatial_stride=spatial_stride,
            )

    return target_path


def prepare_reconstructed_hdf5_output(
    hdf5_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    source_path = Path(hdf5_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if source_path == target_path:
        raise ValueError("Refusing to overwrite the source HDF5 file in place.")
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output HDF5 already exists: {target_path}. "
            "Pass overwrite=True or choose a new path."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    shutil.copy2(source_path, target_path)
    return target_path


def write_reconstructed_record_to_hdf5(
    target: h5py.File,
    record: PDEBenchRecord,
    field_keys: Sequence[str],
    time_slice: slice | None = None,
    spatial_stride: int = 1,
) -> None:
    if spatial_stride <= 0:
        raise ValueError(f"spatial_stride must be positive, got {spatial_stride}")
    for key in field_keys:
        if key not in target or not isinstance(target[key], h5py.Dataset):
            raise KeyError(f"HDF5 dataset key {key!r} not found in copied output.")

    if tuple(record.field_names) != tuple(field_keys):
        raise ValueError(
            "Record field_names must match the export field_keys. "
            f"Got record={record.field_names}, export={tuple(field_keys)}."
        )
    reconstructed = record.reconstructed.detach().cpu()
    if reconstructed.ndim < 3:
        raise ValueError(
            "Expected reconstructed PDEBench tensor shaped "
            f"[spatial..., time, channel], got {tuple(reconstructed.shape)}."
        )
    if int(reconstructed.shape[-1]) != len(field_keys):
        raise ValueError(
            f"Reconstructed tensor has {reconstructed.shape[-1]} channels, "
            f"but field_keys has {len(field_keys)} entries."
        )

    for channel_index, key in enumerate(field_keys):
        dataset = target[key]
        indexer = build_sample_indexer(
            dataset.ndim,
            record.sample_index,
            time_slice,
            spatial_stride,
        )
        field_data = reconstructed[..., channel_index].numpy()
        dataset_payload = np.moveaxis(field_data, -1, 0).astype(dataset.dtype, copy=False)
        expected_shape = hdf5_selection_shape(dataset.shape, indexer)
        if tuple(dataset_payload.shape) != expected_shape:
            raise ValueError(
                f"Reconstructed data for field {key!r} has shape "
                f"{tuple(dataset_payload.shape)}, but target slice shape is "
                f"{expected_shape}."
            )
        dataset[tuple(indexer)] = dataset_payload


def hdf5_selection_shape(
    shape: Sequence[int],
    indexer: Sequence[int | slice],
) -> tuple[int, ...]:
    selected_shape: list[int] = []
    for dim_size, item in zip(shape, indexer):
        if isinstance(item, int):
            continue
        start, stop, step = item.indices(int(dim_size))
        selected_shape.append(len(range(start, stop, step)))
    return tuple(selected_shape)


def pdebench_to_chw_frames(data: torch.Tensor) -> torch.Tensor:
    if data.ndim < 4:
        raise ValueError(
            f"Expected PDEBench data shaped [spatial..., time, channel], got {tuple(data.shape)}"
        )
    if data.ndim != 4:
        raise NotImplementedError(
            "Current AE reconstruction helper supports 2D spatial PDEBench data only. "
            f"Got {data.ndim - 2} spatial dimensions."
        )
    height, width, time_steps, channels = data.shape
    frames = data.permute(2, 3, 0, 1).reshape(time_steps, channels, height, width)
    return frames.contiguous()


def chw_frames_to_pdebench(
    frames: torch.Tensor,
    spatial_shape: Sequence[int],
    time_steps: int,
) -> torch.Tensor:
    if len(spatial_shape) != 2:
        raise NotImplementedError("Only 2D spatial frame reconstruction is currently supported.")
    height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
    channels = int(frames.shape[1])
    return frames.reshape(time_steps, channels, height, width).permute(2, 3, 0, 1).contiguous()


def resize_chw_batch(frames: torch.Tensor, input_size: Sequence[int]) -> torch.Tensor:
    target_size = tuple(int(dim) for dim in input_size)
    if tuple(frames.shape[-2:]) == target_size:
        return frames
    return F.interpolate(frames, size=target_size, mode="bilinear", align_corners=False)


def evaluate_records(
    records: Sequence[PDEBenchRecord],
    operators: Mapping[str, Callable[[dict[str, Any]], Any]],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    aggregate: dict[str, list[float]] = {}
    for record in records:
        direct_metrics = prefix_metrics(
            compute_reconstruction_metrics(
                record.reconstructed.unsqueeze(0),
                record.original.unsqueeze(0),
            ),
            "reconstruction",
        )
        sample_payload: dict[str, Any] = {
            "sample_index": record.sample_index,
            "field_names": list(record.field_names),
            "metrics": direct_metrics,
        }
        add_to_aggregate(aggregate, direct_metrics)

        for name, operator in operators.items():
            original_output = operator(record.as_payload("original"))
            reconstructed_output = operator(record.as_payload("reconstructed"))
            operator_metrics = prefix_metrics(
                compare_outputs(original_output, reconstructed_output),
                name,
            )
            sample_payload["metrics"].update(operator_metrics)
            add_to_aggregate(aggregate, operator_metrics)
        samples.append(sample_payload)

    return {
        "samples": samples,
        "summary": summarize_metrics(aggregate),
    }


def compare_outputs(original_output: Any, reconstructed_output: Any) -> dict[str, float]:
    original_tensor = output_to_tensor(original_output)
    reconstructed_tensor = output_to_tensor(reconstructed_output)
    return compute_reconstruction_metrics(reconstructed_tensor, original_tensor)


def output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.detach().cpu().float()
    if isinstance(output, np.ndarray):
        return torch.as_tensor(output, dtype=torch.float32)
    if isinstance(output, Mapping):
        if "output" in output:
            return output_to_tensor(output["output"])
        if "prediction" in output:
            return output_to_tensor(output["prediction"])
        if "data" in output:
            return output_to_tensor(output["data"])
        tensors = [output_to_tensor(value) for value in output.values() if is_tensor_like(value)]
        if tensors:
            return torch.cat([tensor.reshape(-1) for tensor in tensors])
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        tensors = [output_to_tensor(value) for value in output if is_tensor_like(value)]
        if tensors:
            return torch.cat([tensor.reshape(-1) for tensor in tensors])
    raise TypeError(f"Cannot convert operator output to a tensor: {type(output)!r}")


def is_tensor_like(value: Any) -> bool:
    return isinstance(value, (torch.Tensor, np.ndarray, Mapping, list, tuple))


def prefix_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def add_to_aggregate(aggregate: dict[str, list[float]], metrics: Mapping[str, float]) -> None:
    for key, value in metrics.items():
        if math.isfinite(float(value)):
            aggregate.setdefault(key, []).append(float(value))


def summarize_metrics(aggregate: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key, values in aggregate.items():
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "count": int(array.size),
        }
    return summary


def build_operator(
    operator_type: str,
    *,
    spec: str | None = None,
    checkpoint_path: str | Path | None = None,
    pdebench_root: str | Path | None = None,
    num_channels: int | None = None,
    initial_step: int = 10,
    t_train: int | None = None,
    modes: int = 12,
    width: int = 20,
    init_features: int = 32,
    device: str | torch.device = "auto",
) -> Callable[[dict[str, Any]], Any]:
    operator_type = operator_type.lower()
    if operator_type == "callable":
        if spec is None:
            raise ValueError("--operator-spec is required for operator-type=callable")
        return ExternalCallableOperator(spec, device=device)
    if operator_type == "pdebench-fno":
        if checkpoint_path is None or pdebench_root is None or num_channels is None:
            raise ValueError(
                "pdebench-fno requires checkpoint_path, pdebench_root, and num_channels."
            )
        return PDEBenchFNOForwardOperator(
            pdebench_root=pdebench_root,
            checkpoint_path=checkpoint_path,
            num_channels=num_channels,
            initial_step=initial_step,
            modes=modes,
            width=width,
            t_train=t_train,
            device=device,
        )
    if operator_type == "pdebench-unet":
        if checkpoint_path is None or pdebench_root is None or num_channels is None:
            raise ValueError(
                "pdebench-unet requires checkpoint_path, pdebench_root, and num_channels."
            )
        return PDEBenchUNetForwardOperator(
            pdebench_root=pdebench_root,
            checkpoint_path=checkpoint_path,
            num_channels=num_channels,
            initial_step=initial_step,
            t_train=t_train,
            init_features=init_features,
            device=device,
        )
    raise ValueError(f"Unsupported operator type: {operator_type}")


def load_callable_from_spec(spec: str) -> Callable:
    if ":" not in spec:
        loaded = torch.load(spec, map_location="cpu")
        if not callable(loaded):
            raise TypeError(f"torch.load({spec!r}) did not return a callable object.")
        return loaded
    module_part, attr_name = spec.split(":", maxsplit=1)
    module = load_module(module_part)
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj = obj()
    if not callable(obj):
        raise TypeError(f"Loaded object from {spec!r} is not callable.")
    return obj


def load_module(module_part: str):
    path = Path(module_part)
    if path.suffix == ".py" or path.exists():
        module_path = path.expanduser().resolve()
        module_name = f"_tensor_compression_external_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_part)


def add_pdebench_to_syspath(pdebench_root: str | Path) -> None:
    root = Path(pdebench_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"PDEBench root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def ensure_batched_pdebench_tensor(
    data: torch.Tensor,
    already_batched: bool = False,
) -> torch.Tensor:
    if data.ndim < 4:
        raise ValueError(f"Expected PDEBench tensor [spatial..., time, channel], got {tuple(data.shape)}")
    if not already_batched:
        return data.unsqueeze(0)
    return data


def make_unit_grid(spatial_shape: Sequence[int], device: torch.device) -> torch.Tensor:
    axes = [torch.linspace(0.0, 1.0, int(size), device=device) for size in spatial_shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).unsqueeze(0)


def parse_indices(raw: str | Sequence[int] | None, default: Sequence[int] = (0,)) -> list[int]:
    if raw is None:
        return [int(item) for item in default]
    if isinstance(raw, str):
        if not raw.strip():
            return [int(item) for item in default]
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(item) for item in raw]


def parse_fields(raw: str | Sequence[str] | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
        return values or None
    return [str(item) for item in raw]


def resolve_checkpoint_field_keys(config: Mapping[str, Any] | None) -> list[str] | None:
    if not isinstance(config, Mapping):
        return None
    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        return None
    dataset_cfg = data_cfg.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        return None

    multi_keys = parse_fields(dataset_cfg.get("hdf5_dataset_keys"))
    if multi_keys:
        return multi_keys
    single_key = dataset_cfg.get("hdf5_dataset_key") or dataset_cfg.get("field_key")
    if single_key:
        return [str(single_key)]
    return None


def validate_checkpoint_field_keys_against_model(
    config: Mapping[str, Any] | None,
    field_keys: Sequence[str] | None,
) -> None:
    if not isinstance(config, Mapping) or not field_keys:
        return
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        return
    in_channels = model_cfg.get("in_channels")
    if in_channels is None:
        return
    if int(in_channels) != len(field_keys):
        raise ValueError(
            "Checkpoint field order is inconsistent with model.in_channels. "
            f"field_keys={list(field_keys)}, in_channels={in_channels}."
        )


def resolve_field_keys_for_evaluation(
    *,
    cli_field_keys: Sequence[str] | None,
    checkpoint_field_keys: Sequence[str] | None,
    discovered_field_keys: Sequence[str],
) -> list[str]:
    if checkpoint_field_keys:
        checkpoint_resolved = [str(item) for item in checkpoint_field_keys]
        if cli_field_keys is None:
            return checkpoint_resolved
        cli_resolved = [str(item) for item in cli_field_keys]
        if cli_resolved != checkpoint_resolved:
            raise ValueError(
                "The provided --fields order does not match the compressor checkpoint field order. "
                f"CLI={cli_resolved}, checkpoint={checkpoint_resolved}. "
                "To avoid channel-order mistakes, either omit --fields and use the checkpoint order "
                "automatically, or pass the exact same order as training."
            )
        return checkpoint_resolved
    if cli_field_keys is not None:
        return [str(item) for item in cli_field_keys]
    return [str(item) for item in discovered_field_keys]


def parse_time_slice(start: int | None, stop: int | None, step: int | None = None) -> slice | None:
    if start is None and stop is None and step is None:
        return None
    return slice(start, stop, step)
