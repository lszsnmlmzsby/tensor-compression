from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any


class WandbLogger:
    def __init__(self, config: dict, run_dir: Path) -> None:
        self.enabled = bool(config["wandb"]["enabled"])
        self.run = None
        self._wandb = None
        if not self.enabled:
            return

        import wandb

        wandb_cfg = config["wandb"]
        self._wandb = wandb
        api_key = wandb_cfg.get("api_key") or os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
        self.run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags"),
            config=self._redacted_config(config),
            name=config["experiment"]["name"],
            dir=str(run_dir),
            mode=wandb_cfg.get("mode", "offline"),
        )

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        self._wandb.log(payload, step=step)

    def image(self, data, caption: str | None = None):
        if self.run is None:
            return None
        return self._wandb.Image(data, caption=caption)

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()

    def _redacted_config(self, config: dict) -> dict:
        redacted = copy.deepcopy(config)
        wandb_cfg = redacted.get("wandb", {})
        if wandb_cfg.get("api_key"):
            wandb_cfg["api_key"] = "***REDACTED***"
        return redacted
