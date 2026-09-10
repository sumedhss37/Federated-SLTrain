"""Small, self-contained Muon optimizer for client-side experiments.

Muon is applied only to 2-D matrix parameters.  The implementation keeps the
momentum buffer in FP32 and performs Newton-Schulz orthogonalization in FP32,
while parameter updates are cast back to the parameter dtype.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Muon optimizer for matrix-shaped parameters.

    This implementation intentionally stays in-repo so the client-side Muon
    behavior is easy to inspect and reproduce.  It supports the usual Muon
    knobs: learning rate, momentum, Nesterov, Newton-Schulz steps, and weight
    decay.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if ns_steps < 1:
            raise ValueError("ns_steps must be >= 1")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon client optimizer can only optimize 2-D parameters; "
                        f"got shape={tuple(p.shape)}"
                    )

    @staticmethod
    def _newton_schulz5(x: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
        """Approximate the polar factor using a memory-efficient quintic NS iteration.

        The previous implementation formed ``x @ x.T`` and then ``(x @ x.T) @
        (x @ x.T) @ x``.  For SLTrain factors this can create enormous temporary
        tensors because the larger dimension can be the model hidden dimension.
        Here we always form the Gram matrix on the *smaller* dimension, so the
        expensive square matrices are at most ``min(m, n) x min(m, n)``.
        """
        x = x.float()
        norm = x.norm().clamp_min(eps)
        x = x / norm

        # Standard Muon quintic coefficients.
        a, b, c = 3.4445, -4.7750, 2.0315

        m, n = x.shape
        # Put the smaller dimension in Gram so all matrix products are small.
        use_left_gram = m <= n

        for _ in range(steps):
            if use_left_gram:
                # G: [m, m], m <= n
                G = x @ x.t()
                G2 = G @ G
                x = a * x + b * (G @ x) + c * (G2 @ x)
            else:
                # G: [n, n], n < m
                G = x.t() @ x
                G2 = G @ G
                x = a * x + b * (x @ G) + c * (x @ G2)

        # KellerJordan/original Muon width scaling. The scale depends on the
        # Linear fan-out/fan-in orientation: sqrt(max(1, rows / cols)).
        # Do NOT use the symmetric max/min ratio here: that would also scale
        # wide matrices, which is not the original adjustment.
        x = x * (max(1.0, m / max(1, n))) ** 0.5
        return x

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().float()

                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(grad, dtype=torch.float32)
                    state["momentum_buffer"] = buf

                buf.mul_(momentum).add_(grad)
                update = grad + momentum * buf if nesterov else buf
                update = self._newton_schulz5(update, ns_steps, eps)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)

        return loss
