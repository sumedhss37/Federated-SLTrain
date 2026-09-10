from types import SimpleNamespace

import torch
from torch import nn

from sltrain.client_optim import build_client_optimizer


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.L = nn.Parameter(torch.randn(4, 2))
        self.layer.R = nn.Parameter(torch.randn(2, 4))
        self.other = nn.Parameter(torch.randn(3))


def test_muon_routes_only_sltrain_factors():
    model = Tiny()
    cfg = SimpleNamespace(
        client_optimizer="muon",
        client_lr=1e-3,
        client_weight_decay=0.0,
        muon_lr=0.01,
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_ns_steps=3,
        muon_weight_decay=0.0,
    )
    bundle = build_client_optimizer(model, cfg)
    assert bundle.muon_names == {"layer.L", "layer.R"}
    assert bundle.adamw_names == {"other"}

    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    bundle.step()

    assert not torch.allclose(before["layer.L"], model.layer.L.detach())
    assert not torch.allclose(before["layer.R"], model.layer.R.detach())
    assert not torch.allclose(before["other"], model.other.detach())
