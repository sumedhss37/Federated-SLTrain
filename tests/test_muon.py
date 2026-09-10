import torch

from sltrain.muon import Muon


def test_muon_updates_matrix():
    p = torch.nn.Parameter(torch.randn(8, 4))
    before = p.detach().clone()
    opt = Muon([p], lr=0.01, momentum=0.9, ns_steps=3)
    p.grad = torch.randn_like(p)
    opt.step()
    assert not torch.allclose(before, p.detach())
    assert "momentum_buffer" in opt.state[p]
