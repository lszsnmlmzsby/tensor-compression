from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from tensor_compression.config import load_config
from tensor_compression.downstream.tensor_edit_dataset import (
    TensorEditJsonlDataset,
    tensor_edit_collate_fn,
)
from tensor_compression.integrations import WandbLogger
from tensor_compression.losses.composite import CompositeReconstructionLoss
from tensor_compression.metrics import compute_reconstruction_metrics
from tensor_compression.models import build_model
from tensor_compression.models.editors import EDITOR_REGISTRY
from tensor_compression.models.editors.conditional_tensor_editor_2d import (
    infer_latent_editor_config_from_compressor,
)
from tensor_compression.utils import dump_json, dump_yaml, save_checkpoint, seed_everything


class TensorEditorTrainer:
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.device = self._build_device()
        self.run_dir = self._build_run_dir()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics_latest.json"
        self.wandb_logger = WandbLogger(config=self.config, run_dir=self.run_dir)
        seed_everything(int(self.config["experiment"]["seed"]))
        dump_yaml(self.run_dir / "config_resolved.yaml", self._redacted_config())

    def _build_device(self) -> torch.device:
        requested = str(self.config["experiment"]["device"]).lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _build_run_dir(self) -> Path:
        root = Path(self.config["experiment"]["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{timestamp}_{self.config['experiment']['name']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def validate_setup(self) -> None:
        try:
            dataloaders = self._build_dataloaders()
            model = self._build_model().to(self.device)
            sizes = {split: len(loader.dataset) for split, loader in dataloaders.items()}
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_parameters = sum(p.numel() for p in model.parameters())
            dump_json(
                self.run_dir / "setup_summary.json",
                {
                    "dataset_sizes": sizes,
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "device": str(self.device),
                },
            )
        finally:
            self.wandb_logger.finish()

    def fit(self) -> None:
        dataloaders = self._build_dataloaders()
        if len(dataloaders["train"].dataset) == 0:
            raise RuntimeError("train dataset is empty.")
        if len(dataloaders["val"].dataset) == 0:
            raise RuntimeError("val dataset is empty. Increase data.validation_ratio or provide more records.")

        model = self._build_model().to(self.device)
        criterion = CompositeReconstructionLoss(self.config)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        amp_enabled = bool(self.config["training"]["mixed_precision"]) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_val_loss = float("inf")
        all_metrics: dict[str, dict] = {}
        epochs = int(self.config["training"]["epochs"])
        train_steps_per_epoch = max(1, len(dataloaders["train"]))

        try:
            for epoch in range(1, epochs + 1):
                train_metrics = self._run_epoch(
                    model=model,
                    criterion=criterion,
                    dataloader=dataloaders["train"],
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                )
                val_metrics, val_examples = self._run_validation(
                    model=model,
                    criterion=criterion,
                    dataloader=dataloaders["val"],
                    epoch=epoch,
                )
                if scheduler is not None:
                    scheduler.step()

                merged = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                all_metrics[f"epoch_{epoch:04d}"] = merged
                dump_json(self.metrics_path, all_metrics)
                dump_json(self.run_dir / "val_examples_latest.json", {"examples": val_examples})

                log_step = epoch * train_steps_per_epoch
                self._log_epoch_to_wandb(merged, step=log_step)

                if val_metrics["loss_total"] < best_val_loss:
                    best_val_loss = val_metrics["loss_total"]
                    save_checkpoint(
                        path=self.checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_metric=best_val_loss,
                        config=self._redacted_config(),
                    )

                save_checkpoint(
                    path=self.checkpoint_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_val_loss,
                    config=self._redacted_config(),
                )

                completed_steps = epoch * train_steps_per_epoch
                if completed_steps <= 0:
                    raise RuntimeError("No training steps were completed.")
        finally:
            self.wandb_logger.finish()

    def _build_dataloaders(self) -> dict[str, DataLoader]:
        data_cfg = self.config["editor"]["data"]
        dataset = TensorEditJsonlDataset(
            jsonl_path=data_cfg["jsonl_path"],
            input_size=tuple(int(x) for x in data_cfg.get("input_size", [512, 512])),
            channels=int(data_cfg.get("channels", 1)),
            fix_prompt_mojibake=bool(data_cfg.get("fix_prompt_mojibake", False)),
        )
        val_ratio = float(data_cfg.get("validation_ratio", 0.1))
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("editor.data.validation_ratio must be between 0 and 1.")
        total = len(dataset)
        val_size = max(1, int(math.ceil(total * val_ratio)))
        train_size = total - val_size
        if train_size <= 0:
            raise ValueError("JSONL dataset is too small to create a non-empty train split.")
        generator = torch.Generator().manual_seed(int(self.config["experiment"]["seed"]))
        indices = torch.randperm(total, generator=generator).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        loader_cfg = data_cfg.get("loader", {})
        batch_size = int(loader_cfg.get("batch_size", 1))
        num_workers = int(loader_cfg.get("num_workers", 0))
        persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
        pin_memory = bool(loader_cfg.get("pin_memory", True))
        return {
            "train": DataLoader(
                Subset(dataset, train_indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=tensor_edit_collate_fn,
            ),
            "val": DataLoader(
                Subset(dataset, val_indices),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=tensor_edit_collate_fn,
            ),
        }

    def _build_model(self):
        compressor, compressor_cfg = self._load_compressor()
        data_channels = int(self.config["editor"]["data"].get("channels", 1))
        compressor_in_channels = int(compressor_cfg["model"].get("in_channels", data_channels))
        if compressor_in_channels != data_channels:
            raise ValueError(
                "Tensor editor input channels must match the pretrained compressor. "
                f"Got editor.data.channels={data_channels}, "
                f"compressor model.in_channels={compressor_in_channels}. "
                "For the MVP JSONL format [1, 512, 512], use a 1-channel AE checkpoint."
            )
        data_input_size = tuple(int(x) for x in self.config["editor"]["data"].get("input_size", [512, 512]))
        compressor_input_size = tuple(int(x) for x in compressor_cfg["model"].get("input_size", data_input_size))
        if compressor_input_size != data_input_size:
            raise ValueError(
                "Tensor editor input size must match the pretrained compressor. "
                f"Got editor.data.input_size={data_input_size}, "
                f"compressor model.input_size={compressor_input_size}."
            )
        configured_latent_grid = tuple(int(x) for x in self.config["editor"]["model"].get("latent_grid", []))
        compressor_latent_grid = tuple(int(x) for x in compressor_cfg["model"].get("latent_grid", configured_latent_grid))
        if configured_latent_grid and configured_latent_grid != compressor_latent_grid:
            raise ValueError(
                "editor.model.latent_grid must match the pretrained compressor. "
                f"Got editor.model.latent_grid={configured_latent_grid}, "
                f"compressor model.latent_grid={compressor_latent_grid}."
            )
        editor_name = self.config["editor"]["model"]["name"]
        editor_cls = EDITOR_REGISTRY.get(editor_name)
        self.config["editor"]["model"] = infer_latent_editor_config_from_compressor(
            compressor=compressor,
            base_model_cfg=compressor_cfg["model"],
            editor_model_cfg=self.config["editor"]["model"],
        )
        return editor_cls(compressor=compressor, config=self.config)

    def _load_compressor(self):
        compressor_cfg = self._load_compressor_config()
        compressor = build_model(compressor_cfg)
        checkpoint_path = Path(self.config["editor"]["compressor"]["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Compressor checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = compressor.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Failed to load compressor checkpoint strictly enough for editor training. "
                f"Missing keys: {missing}; unexpected keys: {unexpected}."
            )
        compressor.eval()
        return compressor, compressor_cfg

    def _load_compressor_config(self) -> dict:
        compressor_cfg = self.config["editor"]["compressor"]
        checkpoint_path = Path(compressor_cfg["checkpoint_path"])
        explicit_config_path = compressor_cfg.get("config_path")
        if explicit_config_path:
            return load_config(explicit_config_path, base_root=self.project_root)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Compressor checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        embedded_config = checkpoint.get("config")
        if not isinstance(embedded_config, dict):
            raise ValueError(
                "Compressor checkpoint does not contain an embedded config. "
                "Set editor.compressor.config_path explicitly."
            )
        return embedded_config

    def _build_optimizer(self, model):
        optimizer_cfg = self.config["optimizer"]
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable editor parameters found.")
        kwargs = {
            "lr": float(optimizer_cfg["lr"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
        }
        name = str(optimizer_cfg["name"]).lower()
        if name == "adamw":
            return AdamW(parameters, **kwargs)
        if name == "adam":
            return Adam(parameters, **kwargs)
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")

    def _build_scheduler(self, optimizer):
        scheduler_cfg = self.config["scheduler"]
        name = str(scheduler_cfg["name"]).lower()
        if name == "none":
            return None
        if name == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg["t_max"]),
                eta_min=float(scheduler_cfg["min_lr"]),
            )
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")

    def _redacted_config(self) -> dict:
        redacted = copy.deepcopy(self.config)
        wandb_cfg = redacted.get("wandb", {})
        if wandb_cfg.get("api_key"):
            wandb_cfg["api_key"] = "***REDACTED***"
        return redacted

    def _build_train_step_wandb_payload(self, metrics: dict[str, float]) -> dict[str, float]:
        keep_order = [
            "loss_total",
            "loss_mse",
            "loss_l1",
            "loss_gradient",
            "loss_latent_mse",
            "editor_mae",
            "base_mae",
            "input_mae",
            "gain_vs_input_mae",
            "gain_vs_base_mae",
            "mae_reduction_vs_input",
            "mae_reduction_vs_base",
            "latent_delta_rms",
            "latent_edit_ratio",
            "psnr",
        ]
        return {
            f"train_step/{key}": float(metrics[key])
            for key in keep_order
            if key in metrics and isinstance(metrics[key], (int, float))
        }

    def _log_epoch_to_wandb(self, merged: dict[str, Any], step: int) -> None:
        payload: dict[str, float] = {"epoch": float(merged["epoch"]), "lr": float(merged["lr"])}
        payload.update(self._flatten_metrics("train", merged["train"]))
        payload.update(self._flatten_metrics("val", merged["val"]))
        self.wandb_logger.log(payload, step=step)

    def _flatten_metrics(self, prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
        flattened: dict[str, float] = {}
        for key, value in metrics.items():
            metric_key = f"{prefix}/{key}"
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(metric_key, value))
            elif isinstance(value, bool):
                continue
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                flattened[metric_key] = float(value)
        return flattened

    def _run_epoch(
        self,
        model,
        criterion,
        dataloader,
        optimizer,
        scaler,
        epoch: int,
    ) -> dict[str, float]:
        model.train()
        if getattr(model, "freeze_compressor", False):
            model.compressor.eval()
        running = _MetricAccumulator()
        progress = tqdm(dataloader, desc=f"Epoch {epoch:03d} [editor-train]", leave=False)
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            prompts = batch["prompt"]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                outputs = model(inputs, prompts)
                loss_dict = criterion(outputs["reconstruction"], targets)
                loss_dict = self._add_latent_loss(model, outputs, targets, loss_dict)
            scaler.scale(loss_dict["total"]).backward()
            if self.config["training"]["grad_clip_norm"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(self.config["training"]["grad_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()

            step_metrics = self._build_step_metrics(
                loss_dict,
                outputs,
                inputs.detach(),
                targets.detach(),
            )
            running.update(step_metrics, weight=inputs.shape[0])
            averages = running.averages()
            progress.set_postfix(loss=f"{averages['loss_total']:.4f}", psnr=f"{averages['psnr']:.2f}")
            if step % int(self.config["training"].get("log_interval", 50)) == 0:
                payload = self._build_train_step_wandb_payload(averages)
                if payload:
                    self.wandb_logger.log(
                        payload,
                        step=(epoch - 1) * len(dataloader) + step,
                    )
        return running.averages()

    @torch.no_grad()
    def _run_validation(self, model, criterion, dataloader, epoch: int):
        model.eval()
        running = _MetricAccumulator()
        running_by_type: dict[str, _MetricAccumulator] = {}
        examples: list[dict[str, Any]] = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch:03d} [editor-val]", leave=False)
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            outputs = model(inputs, batch["prompt"])
            loss_dict = criterion(outputs["reconstruction"], targets)
            loss_dict = self._add_latent_loss(model, outputs, targets, loss_dict)
            step_metrics = self._build_step_metrics(loss_dict, outputs, inputs, targets)
            running.update(step_metrics, weight=inputs.shape[0])
            for type_name, type_indices in self._group_batch_indices_by_type(batch["meta"]).items():
                type_metrics = self._build_step_metrics(
                    loss_dict=None,
                    outputs=outputs,
                    inputs=inputs,
                    target=targets,
                    indices=type_indices,
                )
                running_by_type.setdefault(type_name, _MetricAccumulator()).update(
                    type_metrics,
                    weight=len(type_indices),
                )
            averages = running.averages()
            progress.set_postfix(loss=f"{averages['loss_total']:.4f}", psnr=f"{averages['psnr']:.2f}")
            if len(examples) < int(self.config["training"].get("num_saved_val_examples", 8)):
                examples.extend(self._summarize_examples(batch, outputs))
        return (
            {
                **running.averages(),
                "by_type": {
                    type_name: accumulator.averages()
                    for type_name, accumulator in sorted(running_by_type.items())
                },
            },
            examples[: int(self.config["training"].get("num_saved_val_examples", 8))],
        )

    def _build_step_metrics(
        self,
        loss_dict: dict[str, torch.Tensor] | None,
        outputs: dict[str, torch.Tensor],
        inputs: torch.Tensor,
        target: torch.Tensor,
        indices: list[int] | None = None,
    ) -> dict[str, float]:
        if indices is not None:
            original_batch_size = int(target.shape[0])
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=target.device)
            inputs = inputs.index_select(0, index_tensor)
            target = target.index_select(0, index_tensor)
            outputs = self._select_output_batch(outputs, index_tensor, original_batch_size)
        prediction = outputs["reconstruction"].detach()
        base_reconstruction = outputs.get("base_reconstruction")
        if base_reconstruction is not None:
            base_reconstruction = base_reconstruction.detach()
        inputs = inputs.detach()
        target = target.detach()
        metrics = compute_reconstruction_metrics(prediction, target)
        input_metrics = compute_reconstruction_metrics(inputs, target)
        result = {
            **self._prefix_metrics("editor", metrics),
            **self._prefix_metrics("input", input_metrics),
            "gain_vs_input_mse": self._relative_gain(input_metrics["mse"], metrics["mse"]),
            "gain_vs_input_mae": self._relative_gain(input_metrics["mae"], metrics["mae"]),
            "mse_reduction_vs_input": input_metrics["mse"] - metrics["mse"],
            "mae_reduction_vs_input": input_metrics["mae"] - metrics["mae"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "relative_l1": metrics["relative_l1"],
            "max_abs_error": metrics["max_abs_error"],
            "psnr": metrics["psnr"],
        }
        if base_reconstruction is not None:
            base_metrics = compute_reconstruction_metrics(base_reconstruction, target)
            result.update(self._prefix_metrics("base", base_metrics))
            result["gain_vs_base_mse"] = self._relative_gain(base_metrics["mse"], metrics["mse"])
            result["gain_vs_base_mae"] = self._relative_gain(base_metrics["mae"], metrics["mae"])
            result["mse_reduction_vs_base"] = base_metrics["mse"] - metrics["mse"]
            result["mae_reduction_vs_base"] = base_metrics["mae"] - metrics["mae"]
        if "latent_delta" in outputs:
            result["latent_delta_rms"] = self._rms(outputs["latent_delta"].detach())
        if "edited_latent_map" in outputs and "latent_map" in outputs:
            result["latent_edit_ratio"] = self._safe_ratio(
                self._rms_tensor(outputs["edited_latent_map"].detach() - outputs["latent_map"].detach()),
                self._rms_tensor(outputs["latent_map"].detach()),
            )
        if loss_dict is not None:
            result.update(
                {
                    "loss_total": float(loss_dict["total"].detach().cpu().item()),
                    **{
                f"loss_{key}": float(value.detach().cpu().item())
                for key, value in loss_dict.items()
                if key != "total"
                    },
                }
            )
        return result

    def _select_output_batch(
        self,
        outputs: dict[str, torch.Tensor],
        index_tensor: torch.Tensor,
        original_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        selected: dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == original_batch_size:
                selected[key] = value.index_select(0, index_tensor)
        return selected

    def _prefix_metrics(self, prefix: str, metrics: dict[str, float]) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    def _relative_gain(self, baseline: float, edited: float) -> float:
        eps = float(self.config["loss"].get("eps", 1.0e-6))
        if abs(baseline) <= eps:
            return 0.0
        return 1.0 - edited / baseline

    def _rms_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(torch.square(tensor.detach().float())))

    def _rms(self, tensor: torch.Tensor) -> float:
        return float(self._rms_tensor(tensor).detach().cpu().item())

    def _safe_ratio(self, numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        eps = float(self.config["loss"].get("eps", 1.0e-6))
        return float((numerator / torch.clamp(denominator, min=eps)).detach().cpu().item())

    def _group_batch_indices_by_type(self, metas: list[dict[str, Any]]) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for index, meta in enumerate(metas):
            type_name = "unknown"
            if isinstance(meta, dict):
                type_name = str(meta.get("type") or meta.get("perturbation_type") or "unknown")
            grouped.setdefault(type_name, []).append(index)
        return grouped

    def _add_latent_loss(
        self,
        model,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        latent_weight = float(self.config["loss"]["weights"].get("latent_mse", 0.0))
        if latent_weight <= 0.0:
            return loss_dict
        if "edited_latent_map" not in outputs:
            raise KeyError("Editor outputs must include edited_latent_map when loss.weights.latent_mse > 0.")
        target_latent = model.encode_target(targets)
        latent_mse = F.mse_loss(outputs["edited_latent_map"], target_latent["latent_map"])
        merged = dict(loss_dict)
        merged["latent_mse"] = latent_mse
        merged["total"] = merged["total"] + latent_weight * latent_mse
        return merged

    def _summarize_examples(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        prediction_cpu = outputs["reconstruction"].detach().cpu()
        base_cpu = outputs.get("base_reconstruction")
        if base_cpu is not None:
            base_cpu = base_cpu.detach().cpu()
        latent_delta_cpu = outputs.get("latent_delta")
        if latent_delta_cpu is not None:
            latent_delta_cpu = latent_delta_cpu.detach().cpu()
        for row, sample_id in enumerate(batch["sample_id"]):
            pred = prediction_cpu[row]
            target = batch["target"][row]
            inp = batch["input"][row]
            input_mae = float(torch.mean(torch.abs(inp - target)).item())
            prediction_mae = float(torch.mean(torch.abs(pred - target)).item())
            summary = {
                "sample_id": str(sample_id),
                "prompt": batch["prompt"][row],
                "raw_prompt": batch.get("raw_prompt", batch["prompt"])[row],
                "prompt_was_repaired": batch.get("raw_prompt", batch["prompt"])[row] != batch["prompt"][row],
                "meta": batch["meta"][row],
                "type": self._sample_type(batch["meta"][row]),
                "input_mean": float(inp.mean().item()),
                "target_mean": float(target.mean().item()),
                "prediction_mean": float(pred.mean().item()),
                "prediction_min": float(pred.min().item()),
                "prediction_max": float(pred.max().item()),
                "input_mae": input_mae,
                "prediction_mae": prediction_mae,
                "mae": prediction_mae,
                "gain_vs_input_mae": self._relative_gain(input_mae, prediction_mae),
                "mae_reduction_vs_input": input_mae - prediction_mae,
            }
            if base_cpu is not None:
                base = base_cpu[row]
                base_mae = float(torch.mean(torch.abs(base - target)).item())
                summary.update(
                    {
                        "base_mean": float(base.mean().item()),
                        "base_mae": base_mae,
                        "gain_vs_base_mae": self._relative_gain(base_mae, prediction_mae),
                        "mae_reduction_vs_base": base_mae - prediction_mae,
                    }
                )
            if latent_delta_cpu is not None:
                summary["latent_delta_rms"] = float(
                    torch.sqrt(torch.mean(torch.square(latent_delta_cpu[row].float()))).item()
                )
            summaries.append(
                summary
            )
        return summaries

    def _sample_type(self, meta: Any) -> str:
        if isinstance(meta, dict):
            return str(meta.get("type") or meta.get("perturbation_type") or "unknown")
        return "unknown"


class _MetricAccumulator:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.weights: dict[str, float] = {}
        self.sample_count = 0.0

    def update(self, metrics: dict[str, float], weight: int | float) -> None:
        numeric_weight = float(weight)
        if numeric_weight <= 0.0:
            return
        self.sample_count += numeric_weight
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            self.totals[key] = self.totals.get(key, 0.0) + numeric_value * numeric_weight
            self.weights[key] = self.weights.get(key, 0.0) + numeric_weight

    def averages(self) -> dict[str, float]:
        averages = {
            key: self.totals[key] / self.weights[key]
            for key in sorted(self.totals)
            if self.weights.get(key, 0.0) > 0.0
        }
        averages["sample_count"] = self.sample_count
        return averages
