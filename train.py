from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import set_seed as hf_set_seed

from sltrain.client import FederatedClient
from sltrain.config import Config
from sltrain.model import build_model, build_tokenizer
from sltrain.server import FederatedServer
from sltrain.utils import model_parameter_summary, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=Config.model_name)
    p.add_argument("--dataset_name", default=Config.dataset_name)
    p.add_argument("--dataset_config", default=Config.dataset_config)
    p.add_argument("--num_clients", type=int, default=Config.num_clients)
    p.add_argument("--rounds", type=int, default=Config.rounds)
    p.add_argument("--local_steps", type=int, default=Config.local_steps)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--seq_len", type=int, default=Config.seq_len)
    p.add_argument("--rank", type=int, default=Config.rank)
    p.add_argument("--sparsity", type=float, default=Config.sparsity)
    p.add_argument("--lora_alpha", type=float, default=Config.lora_alpha)
    p.add_argument("--client_lr", type=float, default=Config.client_lr)
    p.add_argument("--compression_density", type=float, default=Config.compression_density)
    p.add_argument("--server_lr", type=float, default=Config.server_lr)
    p.add_argument("--server_aggregator", choices=["fedavg", "optimizer"], default=Config.server_aggregator)
    p.add_argument("--server_optimizer", choices=["sgd", "adamw"], default=Config.server_optimizer)
    p.add_argument("--dtype", default=Config.dtype)
    p.add_argument("--output_dir", default=Config.output_dir)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        rank=args.rank,
        sparsity=args.sparsity,
        lora_alpha=args.lora_alpha,
        client_lr=args.client_lr,
        compression_density=args.compression_density,
        server_lr=args.server_lr,
        server_aggregator=args.server_aggregator,
        server_optimizer=args.server_optimizer,
        dtype=args.dtype,
        output_dir=args.output_dir,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save_json(os.path.join(cfg.output_dir, "config.json"))
    set_seed(cfg.seed)
    hf_set_seed(cfg.seed)

    print("Building tokenizer/model...")
    tokenizer = build_tokenizer(cfg)
    model, replaced = build_model(cfg)

    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    print(f"Replaced {len(replaced)} Linear layers with SLLinear")
    print(json.dumps(model_parameter_summary(model), indent=2))

    server = FederatedServer(model, cfg)
    clients = [FederatedClient(i, cfg, model, tokenizer) for i in range(cfg.num_clients)]

    metrics = []
    for r in range(cfg.rounds):
        t0 = time.time()
        payloads = []
        client_losses = []
        executed_steps = []

        for client in clients:
            payload, (loss, nsteps) = client.train_round(server.global_state)
            payloads.append(payload)
            client_losses.append(loss)
            executed_steps.append(nsteps)
            print(f"Round {r:03d} client {client.client_id}: loss={loss:.4f}, steps={nsteps}")

        avg_delta = server.aggregate_payload(payloads)
        elapsed = time.time() - t0
        row = {
            "round": r,
            "mean_client_loss": sum(client_losses) / len(client_losses),
            "client_losses": client_losses,
            "executed_steps": executed_steps,
            "seconds": elapsed,
        }
        metrics.append(row)
        print(f"Round {r:03d} complete: mean_loss={row['mean_client_loss']:.4f}, time={elapsed:.1f}s")

        if (r + 1) % cfg.save_every == 0:
            path = server.save(cfg.output_dir, r + 1)
            print(f"Saved: {path}")

        with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("Federated SLTrain finished.")


if __name__ == "__main__":
    main()
