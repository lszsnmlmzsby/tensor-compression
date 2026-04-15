from __future__ import annotations

import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from tensor_compression.data import build_dataloaders
from tensor_compression.integrations import WandbLogger
from tensor_compression.losses import build_loss
from tensor_compression.metrics import compute_reconstruction_metrics
from tensor_compression.models import build_model
from tensor_compression.utils import (
    build_visualizer,
    dump_json,
    dump_yaml,
    save_checkpoint,
    seed_everything,
)


class CompressionTrainer:
    def __init__(self, config: dict, project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.device = self._build_device()
        self.run_dir = self._build_run_dir()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics_latest.json"
        self.visualizer = build_visualizer(config=self.config, run_dir=self.run_dir)
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
        build_model(self.config)
        build_loss(self.config)
        dataloaders = build_dataloaders(self.config)
        sizes = {split: len(loader.dataset) for split, loader in dataloaders.items()}
        dump_json(self.run_dir / "dataset_sizes.json", sizes)

    def fit(self) -> None:
        model = build_model(self.config).to(self.device)
        criterion = build_loss(self.config)
        dataloaders = build_dataloaders(self.config)
        for split, loader in dataloaders.items():
            if split == "test":
                continue
            if len(loader.dataset) == 0:
                raise RuntimeError(
                    f"{split} dataset is empty. Populate configured data directories before training."
                )
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        amp_enabled = bool(self.config["training"]["mixed_precision"]) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_val_loss = float("inf")
        all_metrics: dict[str, dict] = {}

        epochs = int(self.config["training"]["epochs"])
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(
                model=model,
                criterion=criterion,
                dataloader=dataloaders["train"],
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                training=True,
            )
            val_metrics, val_batch = self._run_validation(
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

            log_payload = {"epoch": epoch, "lr": merged["lr"]}
            log_payload.update({f"train/{k}": v for k, v in train_metrics.items()})
            log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})

            if self.visualizer.should_run(epoch):
                vis_figure = self.visualizer.render(
                    inputs=val_batch["input"],
                    reconstructions=val_batch["reconstruction"],
                )
                vis_path = self.visualizer.save_figure(fig=vis_figure, epoch=epoch)
                image = self.wandb_logger.image(
                    vis_figure,
                    caption=f"epoch={epoch}",
                )
                plt.close(vis_figure)
                if image is not None:
                    log_payload["val/reconstruction"] = image
                    log_payload["val/reconstruction_path"] = str(vis_path)

            self.wandb_logger.log(log_payload, step=epoch)

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

        self.wandb_logger.finish()

    def _build_optimizer(self, model):
        optimizer_cfg = self.config["optimizer"]
        name = str(optimizer_cfg["name"]).lower()
        kwargs = {
            "lr": float(optimizer_cfg["lr"]),
            "weight_decay": float(optimizer_cfg["weight_decay"]),
        }
        if name == "adamw":
            return AdamW(model.parameters(), **kwargs)
        if name == "adam":
            return Adam(model.parameters(), **kwargs)
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")

    def _build_scheduler(self, optimizer):
        scheduler_cfg = self.config["scheduler"]
        name = str(scheduler_cfg["name"]).lower()
        if name == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg["t_max"]),
                eta_min=float(scheduler_cfg["min_lr"]),
            )
        if name == "none":
            return None
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg['name']}")

    def _redacted_config(self) -> dict:
        redacted = copy.deepcopy(self.config)
        wandb_cfg = redacted.get("wandb", {})
        if wandb_cfg.get("api_key"):
            wandb_cfg["api_key"] = "***REDACTED***"
        return redacted

    def _run_epoch(
        self,
        model,
        criterion,
        dataloader,
        optimizer,
        scaler,
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        model.train(training)
        stage = "train" if training else "val"
        running: dict[str, float] = {}
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch:03d} [{stage}]",
            leave=False,
        )

        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=scaler.is_enabled()):
                outputs = model(inputs)
                loss_dict = criterion(outputs["reconstruction"], targets)
            scaler.scale(loss_dict["total"]).backward()
            if self.config["training"]["grad_clip_norm"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(self.config["training"]["grad_clip_norm"]),
                )
            scaler.step(optimizer)
            scaler.update()

            metrics = compute_reconstruction_metrics(outputs["reconstruction"].detach(), targets.detach())
            step_metrics = {
                "loss_total": float(loss_dict["total"].detach().cpu().item()),
                **{f"loss_{k}": float(v.detach().cpu().item()) for k, v in loss_dict.items() if k != "total"},
                **metrics,
            }
            for key, value in step_metrics.items():
                running[key] = running.get(key, 0.0) + value

            averages = {key: value / step for key, value in running.items()}
            progress.set_postfix(
                loss=f"{averages['loss_total']:.4f}",
                psnr=f"{averages['psnr']:.2f}",
            )

            if step % int(self.config["training"]["log_interval"]) == 0:
                self.wandb_logger.log(
                    {f"{stage}_step/{key}": value for key, value in averages.items()},
                    step=(epoch - 1) * len(dataloader) + step,
                )

        return {key: value / max(1, len(dataloader)) for key, value in running.items()}

    @torch.no_grad()
    def _run_validation(self, model, criterion, dataloader, epoch: int):
        model.eval()
        running: dict[str, float] = {}
        first_batch_payload = None
        progress = tqdm(dataloader, desc=f"Epoch {epoch:03d} [val]", leave=False)
        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            outputs = model(inputs)
            loss_dict = criterion(outputs["reconstruction"], targets)
            metrics = compute_reconstruction_metrics(outputs["reconstruction"], targets)
            step_metrics = {
                "loss_total": float(loss_dict["total"].detach().cpu().item()),
                **{f"loss_{k}": float(v.detach().cpu().item()) for k, v in loss_dict.items() if k != "total"},
                **metrics,
            }
            for key, value in step_metrics.items():
                running[key] = running.get(key, 0.0) + value
            averages = {key: value / step for key, value in running.items()}
            progress.set_postfix(
                loss=f"{averages['loss_total']:.4f}",
                psnr=f"{averages['psnr']:.2f}",
            )
            if first_batch_payload is None:
                first_batch_payload = {
                    "input": inputs.detach().cpu(),
                    "reconstruction": outputs["reconstruction"].detach().cpu(),
                }
        return (
            {key: value / max(1, len(dataloader)) for key, value in running.items()},
            first_batch_payload,
        )
