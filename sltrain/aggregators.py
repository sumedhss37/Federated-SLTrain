"""Server-side aggregation strategies.

The key abstraction is that clients produce *pseudo-gradients*:
    delta = theta_start - theta_local_end
which the server interprets as a descent direction. This matches the federated
SLTrain pseudocode and makes the later switch to Muon straightforward.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn


class Aggregator:
    def step(self, global_state: Dict[str, torch.Tensor], avg_delta: Dict[str, torch.Tensor]):
        raise NotImplementedError


class FedAvgAggregator(Aggregator):
    def __init__(self, server_lr: float = 1.0):
        self.server_lr = server_lr

    @torch.no_grad()
    def step(self, global_state, avg_delta):
        return {
            name: global_state[name] - self.server_lr * avg_delta[name]
            for name in global_state
        }


class OptimizerAggregator(Aggregator):
    """Use any torch-style optimizer as the federated server optimizer.

    This is the intended extension point for Muon. The averaged client delta is
    placed into .grad and optimizer.step() is called. Since delta = start - end,
    this has the same sign as a conventional gradient for a descent update.
    """

    def __init__(
        self,
        global_state: Dict[str, torch.Tensor],
        optimizer_factory: Callable,
        lr: float,
        weight_decay: float = 0.0,
        parameter_filter: Optional[Callable[[str, torch.Tensor], bool]] = None,
    ):
        self.names = list(global_state.keys())
        self.params = [nn.Parameter(global_state[n].clone().float()) for n in self.names]
        selected = []
        self.selected_names = []
        for n, p in zip(self.names, self.params):
            if parameter_filter is None or parameter_filter(n, p):
                selected.append(p)
                self.selected_names.append(n)
        self.param_by_name = dict(zip(self.names, self.params))
        self.optimizer = optimizer_factory(selected, lr=lr, weight_decay=weight_decay)

    @torch.no_grad()
    def _sync_params_from_global(self, global_state):
        for name, p in self.param_by_name.items():
            p.copy_(global_state[name])

    @torch.no_grad()
    def step(self, global_state, avg_delta):
        self._sync_params_from_global(global_state)
        self.optimizer.zero_grad(set_to_none=True)
        for name, p in self.param_by_name.items():
            if name in self.selected_names:
                p.grad = avg_delta[name].to(device=p.device, dtype=p.dtype)
        self.optimizer.step()
        return {name: p.detach().cpu().clone() for name, p in self.param_by_name.items()}


class MultiOptimizerAggregator(Aggregator):
    """Apply different optimizer factories to different parameter subsets.

    This is useful for future Muon integration because Muon implementations
    commonly target matrix-valued parameters, while SLTrain sparse values and
    normalization/scalar parameters are vector-valued.
    """

    def __init__(self, global_state, groups: Sequence[Tuple[Callable, Callable, float, float]]):
        self.names = list(global_state.keys())
        self.params = {n: nn.Parameter(global_state[n].clone().float()) for n in self.names}
        self.optimizers = []
        self.group_names = []
        assigned = set()
        for optimizer_factory, predicate, lr, weight_decay in groups:
            names = [n for n in self.names if predicate(n, self.params[n])]
            if not names:
                continue
            if assigned.intersection(names):
                raise ValueError("Server optimizer groups overlap")
            assigned.update(names)
            params = [self.params[n] for n in names]
            self.optimizers.append(optimizer_factory(params, lr=lr, weight_decay=weight_decay))
            self.group_names.append(names)
        if assigned != set(self.names):
            missing = sorted(set(self.names) - assigned)
            raise ValueError(f"Server optimizer groups do not cover parameters, missing: {missing[:5]}")

    @torch.no_grad()
    def step(self, global_state, avg_delta):
        for n in self.names:
            self.params[n].copy_(global_state[n])
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        for names in self.group_names:
            for n in names:
                self.params[n].grad = avg_delta[n].to(device=self.params[n].device, dtype=self.params[n].dtype)
        for opt in self.optimizers:
            opt.step()
        return {n: self.params[n].detach().cpu().clone() for n in self.names}


def make_optimizer_aggregator(
    global_state,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 0.0,
):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        factory = torch.optim.SGD
    elif optimizer_name == "adamw":
        factory = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown server optimizer: {optimizer_name}")
    return OptimizerAggregator(global_state, factory, lr, weight_decay)
