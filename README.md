# Federated SLTrain for Kaggle

This project turns the SLTrain idea into a federated simulation that runs on one Kaggle GPU by executing clients sequentially. It is deliberately split into model, client, server, compression, and aggregation modules so the server optimizer can later be replaced by Muon without rewriting the federated protocol.

## What is implemented

The original SLTrain paper parameterizes every selected linear weight matrix as a low-rank product plus a fixed-support sparse factor, `W = (alpha/r) L R + S`. The sparse support is chosen once at initialization, only sparse values are learned, and the remaining non-linear/non-linear-layer parameters stay full-rank. The paper reports Adam on all linear layers including FC and Q/K/V projections, with the remaining parameters updated full-rank. See the paper for the exact formulation and experimental setup.

The federated pseudocode supplied with this task uses the following round structure:

1. Broadcast global `L/R/S` components.
2. Each client performs local optimization for `H` steps.
3. Compute the client pseudo-gradient `delta = theta_start - theta_end`.
4. Add local error-feedback residual, take Top-k, quantize to 2-bit, and update the residual.
5. Average/decompress the client payloads at the server.
6. Apply a server optimizer to the aggregated pseudo-gradient.

This repository follows that structure. Because real transformer models also contain full-rank parameters outside reparameterized linear layers, the implementation includes those trainable parameters in the same federated update dictionary as an engineering extension.

## Kaggle setup

```python
!git clone <your-github-repo-or-uploaded-repo>
%cd fed_sltrain
!pip install -q -r requirements.txt
```

Or upload the folder/zip and install from the mounted path.

## Smoke test

The defaults are intentionally small enough for a first Kaggle run:

```bash
python -m sltrain.train \
  --model_name EleutherAI/pythia-70m \
  --num_clients 2 \
  --rounds 2 \
  --local_steps 3 \
  --batch_size 2 \
  --seq_len 256 \
  --rank 16 \
  --sparsity 0.03 \
  --compression_density 0.02 \
  --dtype bfloat16
```

For a longer run, increase `rounds` and `local_steps` after verifying the smoke test.

## Switching to Muon later

The server does **not** know or care whether the outer optimizer is SGD, AdamW, or Muon. Clients always return compressed pseudo-gradients and the server receives an averaged dense pseudo-gradient dictionary.

The relevant interface is:

```python
class Aggregator:
    def step(global_state, avg_delta):
        ...
```

`OptimizerAggregator` already implements the adapter pattern expected by a torch-style optimizer: it sets `param.grad = avg_delta[name]` and calls `optimizer.step()`.

So a future Muon integration is isolated to the aggregator construction, for example:

```python
from sltrain.aggregators import OptimizerAggregator
from your_muon_package import Muon

aggregator = OptimizerAggregator(
    server.global_state,
    optimizer_factory=Muon,
    lr=0.02,
    weight_decay=0.0,
)
```

If your Muon implementation only accepts 2-D tensors, `sltrain.muon_hook.make_split_muon_server_aggregator()` provides a ready pattern: Muon is applied to matrix-valued parameters while a fallback optimizer handles sparse values, embeddings/heads, and normalization vectors.

## Files

- `sltrain/layers.py`: custom SLTrain linear layer and replacement logic.
- `sltrain/compression.py`: Top-k + 2-bit compression + error feedback.
- `sltrain/client.py`: persistent client optimizer/error state and local training.
- `sltrain/server.py`: global state and decompression/aggregation loop.
- `sltrain/aggregators.py`: FedAvg and optimizer-backed server aggregation. This is the Muon insertion point.
- `sltrain/data.py`: streaming C4 with deterministic client sharding.
- `sltrain/train.py`: experiment entry point.
- `sltrain/muon_hook.py`: small optional Muon adapter.

## Important caveat about the supplied federated document

The document gives the federated SLTrain protocol and says the communication quantization is 2-bit, but it does not define the exact scalar quantizer. This implementation therefore uses a four-level symmetric 2-bit quantizer with one scale per tensor. That part should be swapped out if you have a reference implementation of SparseLoCo's exact 2-bit codec.
