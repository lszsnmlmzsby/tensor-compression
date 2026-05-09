from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from tensor_compression.config import load_config
from tensor_compression.downstream.tensor_edit_dataset import (
    TensorEditJsonlDataset,
    tensor_edit_collate_fn,
)
from tensor_compression.losses.composite import CompositeReconstructionLoss
from tensor_compression.metrics import compute_reconstruction_metrics
from tensor_compression.models import build_model
from tensor_compression.models.editors import EDITOR_REGISTRY
from tensor_compression.models.editors.conditional_tensor_editor_2d import (
    infer_decoder_config_from_compressor,
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

    def _build_dataloaders(self) -> dict[str, DataLoader]:
        data_cfg = self.config["editor"]["data"]
        dataset = TensorEditJsonlDataset(
            jsonl_path=data_cfg["jsonl_path"],
            input_size=tuple(int(x) for x in data_cfg.get("input_size", [512, 512])),
            channels=int(data_cfg.get("channels", 1)),
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
        self.config["editor"]["model"] = infer_decoder_config_from_compressor(
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
        return copy.deepcopy(self.config)

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
        running: dict[str, float] = {}
        progress = tqdm(dataloader, desc=f"Epoch {epoch:03d} [editor-train]", leave=False)
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            prompts = batch["prompt"]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                outputs = model(inputs, prompts)
                loss_dict = criterion(outputs["reconstruction"], targets)
            scaler.scale(loss_dict["total"]).backward()
            if self.config["training"]["grad_clip_norm"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(self.config["training"]["grad_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()

            step_metrics = self._build_step_metrics(loss_dict, outputs["reconstruction"].detach(), targets.detach())
            for key, value in step_metrics.items():
                running[key] = running.get(key, 0.0) + value
            averages = {key: value / step for key, value in running.items()}
            progress.set_postfix(loss=f"{averages['loss_total']:.4f}", psnr=f"{averages['psnr']:.2f}")
        return {key: value / max(1, len(dataloader)) for key, value in running.items()}

    @torch.no_grad()
    def _run_validation(self, model, criterion, dataloader, epoch: int):
        model.eval()
        running: dict[str, float] = {}
        examples: list[dict[str, Any]] = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch:03d} [editor-val]", leave=False)
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            outputs = model(inputs, batch["prompt"])
            loss_dict = criterion(outputs["reconstruction"], targets)
            step_metrics = self._build_step_metrics(loss_dict, outputs["reconstruction"], targets)
            for key, value in step_metrics.items():
                running[key] = running.get(key, 0.0) + value
            averages = {key: value / step for key, value in running.items()}
            progress.set_postfix(loss=f"{averages['loss_total']:.4f}", psnr=f"{averages['psnr']:.2f}")
            if len(examples) < int(self.config["training"].get("num_saved_val_examples", 8)):
                examples.extend(self._summarize_examples(batch, outputs["reconstruction"]))
        return (
            {key: value / max(1, len(dataloader)) for key, value in running.items()},
            examples[: int(self.config["training"].get("num_saved_val_examples", 8))],
        )

    def _build_step_metrics(
        self,
        loss_dict: dict[str, torch.Tensor],
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float]:
        metrics = compute_reconstruction_metrics(prediction, target)
        return {
            "loss_total": float(loss_dict["total"].detach().cpu().item()),
            **{
                f"loss_{key}": float(value.detach().cpu().item())
                for key, value in loss_dict.items()
                if key != "total"
            },
            **metrics,
        }

    def _summarize_examples(self, batch: dict[str, Any], prediction: torch.Tensor) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        prediction_cpu = prediction.detach().cpu()
        for row, sample_id in enumerate(batch["sample_id"]):
            pred = prediction_cpu[row]
            target = batch["target"][row]
            inp = batch["input"][row]
            summaries.append(
                {
                    "sample_id": str(sample_id),
                    "prompt": batch["prompt"][row],
                    "meta": batch["meta"][row],
                    "input_mean": float(inp.mean().item()),
                    "target_mean": float(target.mean().item()),
                    "prediction_mean": float(pred.mean().item()),
                    "prediction_min": float(pred.min().item()),
                    "prediction_max": float(pred.max().item()),
                    "mae": float(torch.mean(torch.abs(pred - target)).item()),
                }
            )
        return summaries
