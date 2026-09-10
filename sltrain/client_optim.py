"""Client-side optimizer construction for Federated SLTrain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .muon import Muon
from .utils import optimizer_state_to_cpu


@dataclass
class ClientOptimizerBundle:
    """Hybrid client optimizer.

    Muon is used only for SLTrain low-rank factors L/R. All remaining trainable
    parameters use AdamW. The two optimizer states are persisted independently.
    """

    muon: Muon | None
    adamw: torch.optim.AdamW | None
    muon_names: set[str]
    adamw_names: set[str]

    def zero_grad(self):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=True)

    def step(self):
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()

    def state_dict(self):
        return {
            "muon": None if self.muon is None else optimizer_state_to_cpu(self.muon.state_dict()),
            "adamw": None if self.adamw is None else optimizer_state_to_cpu(self.adamw.state_dict()),
        }

    def load_state_dict(self, state):
        if state is None:
            return
        if self.muon is not None and state.get("muon") is not None:
            self.muon.load_state_dict(state["muon"])
        if self.adamw is not None and state.get("adamw") is not None:
            self.adamw.load_state_dict(state["adamw"])


def build_client_optimizer(model, cfg) -> ClientOptimizerBundle:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if cfg.client_optimizer.lower() == "adamw":
        adamw = torch.optim.AdamW(
            [p for _, p in named],
            lr=cfg.client_lr,
            weight_decay=cfg.client_weight_decay,
        )
        return ClientOptimizerBundle(
            muon=None,
            adamw=adamw,
            muon_names=set(),
            adamw_names={n for n, _ in named},
        )

    if cfg.client_optimizer.lower() != "muon":
        raise ValueError(f"Unknown client_optimizer={cfg.client_optimizer}")

    # SLTrain low-rank factors are named *.L and *.R.  Muon is deliberately
    # restricted to these matrix factors rather than embeddings/lm_head/etc.
    muon_named = [(n, p) for n, p in named if n.endswith(".L") or n.endswith(".R")]
    adam_named = [(n, p) for n, p in named if not (n.endswith(".L") or n.endswith(".R"))]

    if not muon_named:
        raise ValueError("Client Muon requested, but no SLTrain .L/.R parameters were found")

    muon = Muon(
        [p for _, p in muon_named],
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
        weight_decay=cfg.muon_weight_decay,
        ns_steps=cfg.muon_ns_steps,
    )

    adamw = None
    if adam_named:
        adamw = torch.optim.AdamW(
            [p for _, p in adam_named],
            lr=cfg.client_lr,
            weight_decay=cfg.client_weight_decay,
        )

    return ClientOptimizerBundle(
        muon=muon,
        adamw=adamw,
        muon_names={n for n, _ in muon_named},
        adamw_names={n for n, _ in adam_named},
    )
