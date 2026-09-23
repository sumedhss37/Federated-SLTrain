import math
import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(name: str):
    name = name.lower()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def model_parameter_summary(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sltrain = 0
    sparse = 0
    lowrank = 0
    for m in model.modules():
        if hasattr(m, "num_lowrank_params"):
            lowrank += m.num_lowrank_params()
            sparse += m.num_sparse_params()
            sltrain += m.num_lowrank_params() + m.num_sparse_params()
    return {
        "total_params": total,
        "trainable_params": trainable,
        "sltrain_params": sltrain,
        "sltrain_lowrank_params": lowrank,
        "sltrain_sparse_params": sparse,
    }


def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def load_trainable_state(model, state: Dict[str, torch.Tensor], device: torch.device):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[name].to(device=device, dtype=p.dtype))


def optimizer_state_to_cpu(state_dict):
    def convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().clone()
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        if isinstance(x, tuple):
            return tuple(convert(v) for v in x)
        return x
    return convert(state_dict)


def scheduled_lr(base_lr: float, min_lr: float, step: int, total_steps: int, warmup_steps: int = 0, schedule: str = "cosine") -> float:
    """Return a per-round learning rate with optional linear warmup and cosine decay."""
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    step = int(step)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unknown lr schedule: {schedule}")

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return min_lr

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine
