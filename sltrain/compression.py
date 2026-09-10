"""SparseLoCo-style top-k + low-bit communication with error feedback.

The federated document specifies: residual accumulation -> Top-k -> 2-bit
quantization -> subtract transmitted/dequantized update from residual.
The exact quantizer is not specified there, so this file uses four symmetric
levels [-1, -1/3, +1/3, +1] with one scale per tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class CompressedTensor:
    shape: tuple
    indices: torch.Tensor  # int64
    codes: torch.Tensor    # uint8, values in [0, 3]
    scale: torch.Tensor    # scalar float tensor


_LEVELS = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)


def _quantize_2bit(values: torch.Tensor):
    scale = values.abs().max().float().clamp_min(1e-12)
    norm = (values.float() / scale).clamp(-1, 1)
    levels = _LEVELS.to(device=norm.device)
    codes = (norm.unsqueeze(-1) - levels.view(1, -1)).abs().argmin(dim=-1).to(torch.uint8)
    return codes, scale.cpu()


def _dequantize_2bit(codes: torch.Tensor, scale: torch.Tensor, device=None, dtype=torch.float32):
    if device is None:
        device = codes.device
    levels = _LEVELS.to(device=device)
    return levels[codes.long()].to(dtype=dtype) * scale.to(device=device, dtype=dtype)


def compress_tensor(delta: torch.Tensor, residual: torch.Tensor, density: float):
    if not 0 < density <= 1:
        raise ValueError("density must be in (0,1]")
    flat = delta.detach().float().cpu().reshape(-1)
    residual.add_(flat)

    k = max(1, int(round(density * residual.numel())))
    k = min(k, residual.numel())
    _, idx = torch.topk(residual.abs(), k=k, largest=True, sorted=False)
    vals = residual[idx]

    codes, scale = _quantize_2bit(vals)
    deq = _dequantize_2bit(codes, scale, device=residual.device, dtype=residual.dtype)
    residual[idx] -= deq

    return CompressedTensor(
        shape=tuple(delta.shape),
        indices=idx.clone(),
        codes=codes.cpu(),
        scale=scale,
    )


def decompress_tensor(payload: CompressedTensor, dtype=torch.float32):
    out = torch.zeros(int(torch.tensor(payload.shape).prod().item()), dtype=dtype)
    vals = _dequantize_2bit(payload.codes, payload.scale, device=out.device, dtype=dtype)
    out[payload.indices] = vals
    return out.reshape(payload.shape)


def compress_state_dict(delta_state: Dict[str, torch.Tensor], residuals: Dict[str, torch.Tensor], density: float):
    payload = {}
    for name, delta in delta_state.items():
        if name not in residuals:
            residuals[name] = torch.zeros(delta.numel(), dtype=torch.float32)
        payload[name] = compress_tensor(delta, residuals[name], density)
    return payload


def decompress_state_dict(payload: Dict[str, CompressedTensor], dtype=torch.float32):
    return {name: decompress_tensor(p, dtype=dtype) for name, p in payload.items()}


def payload_nbytes(payload) -> int:
    """Return the logical bytes required to transmit one client payload.

    For dense tensors this counts their current dtype bytes. For CompressedTensor
    this counts the actual stored representation in this implementation: int64
    indices, uint8 codes, one scalar scale, plus a small shape metadata cost.
    Note that codes are currently stored one-per-byte (uint8), not bit-packed.
    """
    if torch.is_tensor(payload):
        return int(payload.numel() * payload.element_size())

    if isinstance(payload, CompressedTensor):
        index_bytes = int(payload.indices.numel() * payload.indices.element_size())
        code_bytes = int(payload.codes.numel() * payload.codes.element_size())
        scale_bytes = int(payload.scale.numel() * payload.scale.element_size())
        # Approximate fixed serialization metadata: ndim + int64 shape values.
        metadata_bytes = 4 + 8 * len(payload.shape)
        return index_bytes + code_bytes + scale_bytes + metadata_bytes

    raise TypeError(f"Unsupported payload type: {type(payload)}")


def state_payload_nbytes(payload_state: Dict[str, object]) -> int:
    """Total logical transmitted bytes for one client state-dict payload."""
    return sum(payload_nbytes(v) for v in payload_state.values())
