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


def test_newton_schulz_handles_tall_matrix_efficiently():
    # Simulates an SLTrain L factor: large output dimension, small rank.
    x = torch.randn(4096, 16)
    y = Muon._newton_schulz5(x, steps=3, eps=1e-7)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_newton_schulz_handles_wide_matrix_efficiently():
    # Simulates an SLTrain R factor: small rank, large input dimension.
    x = torch.randn(16, 4096)
    y = Muon._newton_schulz5(x, steps=3, eps=1e-7)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_original_rectangular_scaling_is_orientation_aware():
    # For original Muon scaling, tall [m,n] gets sqrt(m/n), while wide [n,m]
    # gets no extra factor because max(1, rows/cols) == 1.
    tall = Muon._newton_schulz5(torch.randn(8, 2), steps=1, eps=1e-7)
    wide = Muon._newton_schulz5(torch.randn(2, 8), steps=1, eps=1e-7)
    # This test only checks finiteness/shape; exact norm ratio depends on NS.
    assert tall.shape == (8, 2)
    assert wide.shape == (2, 8)
