# Federated SLTrain on Kaggle

This project implements a single-GPU federated simulation of SLTrain. Clients are executed sequentially on one GPU. SLTrain replaces selected Linear layers by `W = (alpha/r) L R + S`, with a fixed random support for the sparse component.

## Client optimizer modes

`--client_optimizer adamw` (default): AdamW on all trainable parameters.

`--client_optimizer muon`: client-side Muon on the SLTrain low-rank factor matrices `*.L` and `*.R`, with AdamW on all remaining trainable parameters. This is the implementation of client-side Muon: Muon is applied to each client's local gradients at every local optimization step, before the client's final pseudo-gradient/model delta is formed and sent to the server.

The server remains a plain aggregator (`fedavg` by default). Client optimizer state is persistent across federated rounds and is kept on CPU between client executions.

## Example: client-side Muon, dense communication

```bash
python3 train.py \
  --model_name EleutherAI/pythia-70m \
  --dataset_name allenai/c4 \
  --dataset_config en \
  --num_clients 2 \
  --rounds 5 \
  --local_steps 10 \
  --batch_size 4 \
  --seq_len 256 \
  --rank 16 \
  --sparsity 0.03 \
  --client_optimizer muon \
  --muon_lr 0.02 \
  --muon_momentum 0.95 \
  --muon_ns_steps 5 \
  --compression_mode none
```

## Main knobs

### Federated/data
- `--num_clients`: number of simulated clients.
- `--rounds`: number of outer federated rounds.
- `--local_steps`: local optimizer steps per client per round.
- `--batch_size`: micro-batch size per local step.
- `--seq_len`: tokens per sequence.

### SLTrain
- `--rank`: low-rank factor dimension.
- `--sparsity`: fraction of entries used by the fixed sparse support.
- `--lora_alpha`: scales the low-rank term by `alpha/r`.

### Client optimizer
- `--client_optimizer adamw|muon`: choose the local optimizer.
- `--client_lr`: AdamW learning rate, and fallback LR for non-Muon parameters when client Muon is enabled.
- `--client_weight_decay`: AdamW weight decay.
- `--muon_lr`: Muon learning rate for L/R factors.
- `--muon_momentum`: Muon momentum coefficient.
- `--muon_ns_steps`: Newton-Schulz iterations.
- `--muon_weight_decay`: decoupled weight decay for Muon factors.
- `--no_muon_nesterov`: disable Muon Nesterov momentum.
- `--grad_clip`: global gradient clipping threshold.

### Communication
- `--compression_mode sparse|none`: SparseLoCo-style compressed payload or dense client delta.
- `--compression_density`: Top-k density used when compression is enabled.

### Server
- `--server_aggregator fedavg|optimizer`: server aggregation strategy already present in the project.
- `--server_lr`: server learning rate.
- `--server_optimizer sgd|adamw`: optimizer used by the generic optimizer-based server aggregator.

### Precision/runtime
- `--dtype`: `bfloat16`, `float16`, or `float32`.
- `--no_gradient_checkpointing`: disable activation checkpointing.

### Global validation
- `--eval_every`: evaluate the aggregated global model every N outer rounds.
- `--eval_batches`: number of fixed C4 validation batches used for each evaluation.

The validation batches are prepared once from the C4 validation split and reused
after every evaluation round, so `global/val_loss` and `global/val_perplexity` are
directly comparable across rounds. These are measured on the actual aggregated
global model after the server update.

### W&B
- `--wandb_project`: W&B project.
- `--wandb_entity`: optional entity.
- `--wandb_run_name`: optional run name.
- `--no_wandb`: disable W&B.

## Interpretation of client-side Muon

For each local step, a client computes the ordinary gradient `g` from its current mini-batch. For `*.L` and `*.R`, Muon forms momentum, applies Newton-Schulz orthogonalization, and updates the factors. Other trainable parameters use AdamW. After all local steps, the client forms `delta = theta_start - theta_end`; this is what is aggregated by the server.

The project includes unit tests for SLTrain, compression, and Muon.
