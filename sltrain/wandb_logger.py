from __future__ import annotations

import os
from typing import Any, Dict, Optional

import wandb


class WandbLogger:
    """Small adapter so the training loop stays independent of W&B."""

    def __init__(self, cfg, parameter_summary: Dict[str, Any]):
        self.enabled = bool(cfg.wandb_enabled)
        self.run = None
        if not self.enabled:
            return

        init_kwargs = {
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity or None,
            "name": cfg.wandb_run_name or None,
            "config": {**cfg.to_dict(), **parameter_summary},
            "reinit": "return_previous",
            "job_type": "federated-sltrain",
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        self.run = wandb.init(**init_kwargs)

        if self.run is not None:
            self.run.define_metric("round")
            self.run.define_metric("train/*", step_metric="round")
            self.run.define_metric("client/*", step_metric="round")
            self.run.define_metric("server/*", step_metric="round")
            self.run.define_metric("global/*", step_metric="round")
            self.run.define_metric("system/*", step_metric="round")
            self.run.define_metric("communication/*", step_metric="round")

    def log_round(self, metrics: Dict[str, Any], step: int):
        if self.run is None:
            return
        self.run.log({"round": step, **metrics}, step=step)

    def log_final_model(self, model_path: str):
        if self.run is None:
            return
        artifact = wandb.Artifact(
            name=self.run.name + "-final-model",
            type="model",
            metadata={"checkpoint": "final", "format": "safetensors"},
        )
        artifact.add_file(model_path, name=os.path.basename(model_path))
        self.run.log_artifact(artifact)
        self.run.summary["final_model_artifact"] = artifact.name
        self.run.summary["final_model_path"] = model_path

    def log_summary(self, summary: Dict[str, Any]):
        if self.run is None:
            return
        for key, value in summary.items():
            self.run.summary[key] = value

    def finish(self):
        if self.run is not None:
            wandb.finish()
