import torch
from sltrain.compression import compress_tensor, decompress_tensor


def test_compression_error_feedback_shape():
    x = torch.randn(100)
    residual = torch.zeros(100)
    payload = compress_tensor(x, residual, density=0.1)
    y = decompress_tensor(payload)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert residual.shape == x.shape
