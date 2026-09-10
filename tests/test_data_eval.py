import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None
datasets_available = importlib.util.find_spec("datasets") is not None

pytestmark = pytest.mark.skipif(not datasets_available, reason="datasets package not installed")

if torch_available:
    import torch


class DummyModel(torch.nn.Module):
    def forward(self, input_ids, labels, attention_mask=None, use_cache=False):
        class O:
            pass
        o = O()
        o.loss = torch.tensor(2.0, device=input_ids.device)
        return o


def test_eval_import_and_dummy_model():
    from sltrain.data import evaluate_model

    class Validation:
        batches = [{
            "input_ids": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.ones(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }]

        def __len__(self):
            return 1

    m = DummyModel()
    out = evaluate_model(m, Validation(), torch.device("cpu"))
    assert abs(out["loss"] - 2.0) < 1e-6
    assert out["perplexity"] > 7.0
