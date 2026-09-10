"""Memory-conscious SLTrain linear layer.

The paper parameterizes a dense weight matrix as W = (alpha/r) * L @ R + S,
where S has fixed random support and only the sparse values are learned.
This implementation avoids storing the dense W for autograd by implementing
forward/backward directly in terms of L, R, sparse values, and their indices.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class _SLLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, L, R, sparse_values, rows, cols, scale, bias):
        # x: [..., in_features]
        x2 = x.reshape(-1, x.shape[-1])

        # Low-rank term without materializing L @ R.
        low = F.linear(F.linear(x2, R), L * scale)

        # Sparse S contribution: S[row_j, col_j] = sparse_values[j].
        out = low
        if sparse_values.numel() > 0:
            contrib = x2[:, cols] * sparse_values.unsqueeze(0)
            sparse_out = torch.zeros(
                x2.shape[0], L.shape[0], device=x.device, dtype=x.dtype
            )
            sparse_out.scatter_add_(1, rows.unsqueeze(0).expand(x2.shape[0], -1), contrib)
            out = out + sparse_out

        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x, L, R, sparse_values, rows, cols)
        ctx.scale = scale
        ctx.has_bias = bias is not None
        return out.reshape(*x.shape[:-1], L.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, L, R, sparse_values, rows, cols = ctx.saved_tensors
        scale = ctx.scale

        x2 = x.reshape(-1, x.shape[-1])
        go2 = grad_output.reshape(-1, grad_output.shape[-1])

        # grad wrt input:
        # dL/dx from low-rank term plus sparse selected columns.
        mid = go2 @ (L * scale)                 # [N, rank]
        grad_x = mid @ R                        # [N, in]
        if sparse_values.numel() > 0:
            contrib = go2[:, rows] * sparse_values.unsqueeze(0)
            grad_x = grad_x.scatter_add(
                1,
                cols.unsqueeze(0).expand(x2.shape[0], -1),
                contrib,
            )
        grad_x = grad_x.reshape_as(x)

        # grad L = scale * (go^T @ (x R^T))
        xr = x2 @ R.t()
        grad_L = (go2.t() @ xr) * scale

        # grad R = scale * ((L^T @ go^T) @ x)
        goL = go2 @ (L * scale)
        grad_R = goL.t() @ x2

        # Sparse values are exactly the dense weight gradient evaluated at support.
        grad_sparse = None
        if sparse_values.numel() > 0:
            grad_sparse = (go2[:, rows] * x2[:, cols]).sum(dim=0)

        grad_bias = go2.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_L, grad_R, grad_sparse, None, None, None, grad_bias


class SLLinear(nn.Module):
    """SLTrain replacement for nn.Linear.

    W = scale * L @ R + Sparse(rows, cols, values)
    with fixed random support and learnable sparse values.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        sparsity: float,
        alpha: float,
        bias: bool = False,
        support_seed: int = 42,
    ):
        super().__init__()
        if not 0 < sparsity <= 1:
            raise ValueError("sparsity must be in (0, 1]")
        if rank < 1 or rank >= min(in_features, out_features):
            raise ValueError(
                f"rank={rank} must satisfy 1 <= rank < min(in={in_features}, out={out_features})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.sparsity = sparsity
        self.alpha = alpha
        self.scale = alpha / rank

        self.L = nn.Parameter(torch.empty(out_features, rank))
        self.R = nn.Parameter(torch.empty(rank, in_features))

        nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
        nn.init.zeros_(self.L)

        nnz = max(1, int(round(sparsity * in_features * out_features)))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(support_seed) & 0x7FFFFFFF)
        perm = torch.randperm(in_features * out_features, generator=g)[:nnz]
        rows = torch.div(perm, in_features, rounding_mode="floor").long()
        cols = (perm % in_features).long()
        self.register_buffer("rows", rows, persistent=True)
        self.register_buffer("cols", cols, persistent=True)

        bound = 1.0 / math.sqrt(in_features)
        self.sparse_values = nn.Parameter(torch.empty(nnz))
        nn.init.uniform_(self.sparse_values, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            **kwargs,
        )
        if linear.bias is not None:
            with torch.no_grad():
                layer.bias.copy_(linear.bias.detach())
        return layer

    def forward(self, x):
        return _SLLinearFunction.apply(
            x,
            self.L,
            self.R,
            self.sparse_values,
            self.rows,
            self.cols,
            self.scale,
            self.bias,
        )

    def effective_weight(self):
        """Materialize W only for debugging/evaluation, never used by training."""
        W = self.scale * (self.L @ self.R)
        if self.sparse_values.numel():
            W = W.clone()
            W.index_put_((self.rows, self.cols), self.sparse_values, accumulate=True)
        return W

    def num_lowrank_params(self):
        return self.L.numel() + self.R.numel()

    def num_sparse_params(self):
        return self.sparse_values.numel()


def _stable_name_seed(base_seed: int, name: str) -> int:
    # Stable across Python processes, unlike hash(name).
    h = 2166136261
    for c in name.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return (int(base_seed) + h) & 0x7FFFFFFF


def replace_linear_layers(
    model: nn.Module,
    rank: int,
    sparsity: float,
    alpha: float,
    seed: int = 42,
    exclude_patterns=(),
):
    """Replace selected nn.Linear modules by SLLinear.

    All clients must use the same seed and model module names so the sparse support
    is identical, which is required for straightforward federated aggregation.
    """
    replaced = []

    def visit(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                excluded = any(pat in full_name for pat in exclude_patterns)
                if not excluded:
                    new_layer = SLLinear.from_linear(
                        child,
                        rank=rank,
                        sparsity=sparsity,
                        alpha=alpha,
                        support_seed=_stable_name_seed(seed, full_name),
                    )
                    new_layer = new_layer.to(device=child.weight.device, dtype=child.weight.dtype)
                    setattr(parent, child_name, new_layer)
                    replaced.append(full_name)
                    continue
            visit(child, full_name)

    visit(model)
    return replaced
