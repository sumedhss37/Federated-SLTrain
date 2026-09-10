# Adam and SparseLoCo Training Configurations

This file contains the training commands for the following
configurations:

1.  AdamW
2.  AdamW + SparseLoCo
3.  Muon
4.  Muon + SparseLoCo

All experiments use the same model, dataset, client count, batch size,
sequence length, rank, and sparsity unless otherwise specified.

------------------------------------------------------------------------

## 1. AdamW

``` bash
!python3 train.py \
    --model_name EleutherAI/pythia-70m \
    --dataset_name allenai/c4 \
    --dataset_config en \
    --num_clients 3 \
    --rounds 35 \
    --local_steps 10 \
    --batch_size 4 \
    --seq_len 256 \
    --rank 16 \
    --sparsity 0.03 \
    --client_optimizer adamw \
    --compression_mode none
```

------------------------------------------------------------------------

## 2. AdamW + SparseLoCo

``` bash
!python3 train.py \
    --model_name EleutherAI/pythia-70m \
    --dataset_name allenai/c4 \
    --dataset_config en \
    --num_clients 3 \
    --rounds 50 \
    --local_steps 10 \
    --batch_size 4 \
    --seq_len 256 \
    --rank 16 \
    --sparsity 0.03
```

------------------------------------------------------------------------

## 3. Muon

``` bash
!python3 train.py \
    --model_name EleutherAI/pythia-70m \
    --dataset_name allenai/c4 \
    --dataset_config en \
    --num_clients 3 \
    --rounds 5 \
    --local_steps 10 \
    --batch_size 4 \
    --seq_len 256 \
    --rank 16 \
    --sparsity 0.03 \
    --client_optimizer muon \
    --muon_lr 0.002 \
    --muon_momentum 0.95 \
    --muon_ns_steps 5 \
    --compression_mode none
```

------------------------------------------------------------------------

## 4. Muon + SparseLoCo

``` bash
!python3 train.py \
    --model_name EleutherAI/pythia-70m \
    --dataset_name allenai/c4 \
    --dataset_config en \
    --num_clients 3 \
    --rounds 50 \
    --local_steps 10 \
    --batch_size 4 \
    --seq_len 256 \
    --rank 16 \
    --sparsity 0.03 \
    --client_optimizer muon \
    --muon_lr 0.002 \
    --muon_momentum 0.95 \
    --muon_ns_steps 5 \
    --compression_mode sparse \
    --compression_density 0.02
```
