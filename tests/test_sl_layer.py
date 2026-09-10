import torch
from torch.nn import functional as F

from sltrain.layers import SLLinear


def test_forward_and_backward_against_dense():
    torch.manual_seed(0)
    layer = SLLinear(7, 5, rank=3, sparsity=0.2, alpha=6.0, support_seed=123)
    x = torch.randn(4, 7, requires_grad=True)

    y = layer(x)
    loss = (y ** 2).sum()
    loss.backward()

    # Dense reference with the exact same parameters.
    xd = x.detach().clone().requires_grad_(True)
    W = layer.effective_weight().detach().requires_grad_(True)
    yd = F.linear(xd, W, layer.bias.detach() if layer.bias is not None else None)
    ld = (yd ** 2).sum()
    ld.backward()

    assert torch.allclose(y.detach(), yd.detach(), atol=1e-5, rtol=1e-5)
    # Compare effective-weight gradient at fixed support to L/R parameter grads
    # via finite-difference-like autograd consistency on outputs.
    eps = 1e-3
    idx = 0
    old = layer.sparse_values.detach()[idx].item()
    with torch.no_grad():
        layer.sparse_values[idx] = old + eps
    yp = layer(x.detach())
    with torch.no_grad():
        layer.sparse_values[idx] = old - eps
    ym = layer(x.detach())
    with torch.no_grad():
        layer.sparse_values[idx] = old
    numeric = (((yp ** 2).sum() - (ym ** 2).sum()) / (2 * eps)).item()
    autograd = layer.sparse_values.grad[idx].item()
    assert abs(numeric - autograd) < 1e-2
