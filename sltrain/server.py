from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict

import torch
from safetensors.torch import save_file

from .aggregators import FedAvgAggregator, make_optimizer_aggregator
from .compression import decompress_tensor
from .utils import trainable_state_dict


class FederatedServer:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.global_state = trainable_state_dict(model)
        if cfg.server_aggregator == "fedavg":
            self.aggregator = FedAvgAggregator(cfg.server_lr)
        elif cfg.server_aggregator == "optimizer":
            self.aggregator = make_optimizer_aggregator(
                self.global_state,
                optimizer_name=cfg.server_optimizer,
                lr=cfg.server_lr,
                weight_decay=cfg.server_weight_decay,
            )
        else:
            raise ValueError(f"Unknown server_aggregator={cfg.server_aggregator}")

    def aggregate_payload(self, payloads):
        if not payloads:
            raise ValueError("No client payloads")

        acc = {name: torch.zeros_like(t, dtype=torch.float32) for name, t in self.global_state.items()}
        for payload in payloads:
            for name, packed in payload.items():
                acc[name].add_(decompress_tensor(packed, dtype=torch.float32))

        inv_m = 1.0 / len(payloads)
        avg = {name: t * inv_m for name, t in acc.items()}
        self.global_state = self.aggregator.step(self.global_state, avg)
        return avg

    def load_into_model(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.global_state[name].to(device=p.device, dtype=p.dtype))

    def save(self, out_dir: str, round_idx: int):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"global_round_{round_idx:04d}.safetensors")
        cpu_state = OrderedDict((k, v.contiguous()) for k, v in self.global_state.items())
        save_file(cpu_state, path)
        return path
