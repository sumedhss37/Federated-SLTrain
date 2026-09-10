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
from sltrain.wandb_logger import WandbLogger
from sltrain.data import C4ValidationSet, evaluate_model
from sltrain.compression import state_payload_nbytes


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
    p.add_argument("--client_optimizer", choices=["adamw", "muon"], default=Config.client_optimizer)
    p.add_argument("--client_lr", type=float, default=Config.client_lr)
    p.add_argument("--muon_lr", type=float, default=Config.muon_lr)
    p.add_argument("--muon_momentum", type=float, default=Config.muon_momentum)
    p.add_argument("--muon_ns_steps", type=int, default=Config.muon_ns_steps)
    p.add_argument("--muon_weight_decay", type=float, default=Config.muon_weight_decay)
    p.add_argument("--no_muon_nesterov", action="store_true")
    p.add_argument("--compression_mode", choices=["sparse", "none"], default=Config.compression_mode)
    p.add_argument("--compression_density", type=float, default=Config.compression_density)
    p.add_argument("--server_lr", type=float, default=Config.server_lr)
    p.add_argument("--server_aggregator", choices=["fedavg", "optimizer"], default=Config.server_aggregator)
    p.add_argument("--server_optimizer", choices=["sgd", "adamw"], default=Config.server_optimizer)
    p.add_argument("--dtype", default=Config.dtype)
    p.add_argument("--output_dir", default=Config.output_dir)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--eval_every", type=int, default=Config.eval_every)
    p.add_argument("--eval_batches", type=int, default=Config.eval_batches)

    p.add_argument("--wandb_project", default=Config.wandb_project)
    p.add_argument("--wandb_entity", default=Config.wandb_entity)
    p.add_argument("--wandb_run_name", default=Config.wandb_run_name)
    p.add_argument("--no_wandb", action="store_true")
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
        client_optimizer=args.client_optimizer,
        client_lr=args.client_lr,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_weight_decay=args.muon_weight_decay,
        muon_nesterov=not args.no_muon_nesterov,
        compression_mode=args.compression_mode,
        compression_density=args.compression_density,
        server_lr=args.server_lr,
        server_aggregator=args.server_aggregator,
        server_optimizer=args.server_optimizer,
        dtype=args.dtype,
        output_dir=args.output_dir,
        seed=args.seed,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        wandb_enabled=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
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

    summary = model_parameter_summary(model)
    print(f"Replaced {len(replaced)} Linear layers with SLLinear")
    print(json.dumps(summary, indent=2))

    logger = WandbLogger(cfg, summary)

    validation_set = C4ValidationSet(
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        num_batches=cfg.eval_batches,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
    )
    print(f"Prepared {len(validation_set)} fixed C4 validation batches")

    server = FederatedServer(model, cfg)
    clients = [FederatedClient(i, cfg, model, tokenizer) for i in range(cfg.num_clients)]

    metrics = []
    total_tokens = 0

    try:
        for r in range(cfg.rounds):
            t0 = time.time()
            payloads = []
            client_losses = []
            executed_steps = []

            for client in clients:
                payload, stats = client.train_round(server.global_state)
                loss, nsteps, ntokens = stats
                payloads.append(payload)
                client_losses.append(loss)
                executed_steps.append(nsteps)
                total_tokens += ntokens
                print(
                    f"Round {r:03d} client {client.client_id}: "
                    f"loss={loss:.4f}, steps={nsteps}, tokens={ntokens}"
                )

            # Logical client -> server communication volume for this outer round.
            # This counts the actual payload representation produced by each client,
            # rather than the dense size after server-side decompression.
            client_upload_bytes = [state_payload_nbytes(payload) for payload in payloads]
            total_client_upload_bytes = sum(client_upload_bytes)

            avg_delta = server.aggregate_payload(payloads)
            elapsed = time.time() - t0
            mean_loss = sum(client_losses) / len(client_losses)

            global_eval = None
            eval_elapsed = 0.0
            if cfg.eval_every > 0 and (r + 1) % cfg.eval_every == 0:
                # Evaluate the actual aggregated global model, not the clients'
                # local training losses.
                eval_t0 = time.time()
                server.load_into_model(model)
                global_eval = evaluate_model(
                    model,
                    validation_set,
                    torch.device(cfg.device if torch.cuda.is_available() else "cpu"),
                )
                eval_elapsed = time.time() - eval_t0
                print(
                    f"Round {r:03d} global eval: "
                    f"val_loss={global_eval['loss']:.4f}, "
                    f"val_ppl={global_eval['perplexity']:.2f}"
                )

            # Rough dense communication size for the client payloads. Compression
            # bytes can be added later when the payload codec reports exact size.
            dense_bytes = sum(t.numel() * 4 for t in avg_delta.values())

            row = {
                "round": r,
                "mean_client_loss": mean_loss,
                "client_losses": client_losses,
                "executed_steps": executed_steps,
                "seconds": elapsed,
                "tokens_this_round": sum(executed_steps) * cfg.batch_size * cfg.seq_len,
                "tokens_seen_total": total_tokens,
                "global_val_loss": None if global_eval is None else global_eval["loss"],
                "global_val_perplexity": None if global_eval is None else global_eval["perplexity"],
            }
            metrics.append(row)

            wb_metrics = {
                "train/mean_client_loss": mean_loss,
                "train/mean_client_perplexity": float(torch.exp(torch.tensor(mean_loss))),
                "train/tokens_this_round": row["tokens_this_round"],
                "train/tokens_seen_total": total_tokens,
                "server/avg_delta_l2": float(torch.sqrt(sum((v.float() ** 2).sum() for v in avg_delta.values()))),
                "server/avg_delta_abs_mean": float(torch.cat([v.float().reshape(-1) for v in avg_delta.values()]).abs().mean()),
                "system/round_seconds": elapsed,
                "system/eval_seconds": eval_elapsed,
                "system/max_cuda_memory_gb": (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available() else 0.0
                ),
                "communication/dense_avg_delta_mb": dense_bytes / (1024 ** 2),
                "communication/client_to_server_bytes": total_client_upload_bytes,
                "communication/client_to_server_mb": total_client_upload_bytes / (1024 ** 2),
                "communication/client_to_server_avg_mb": (
                    (total_client_upload_bytes / len(payloads)) / (1024 ** 2)
                    if payloads else 0.0
                ),
                "communication/client_to_server_compression_ratio": (
                    (dense_bytes * len(payloads)) / total_client_upload_bytes
                    if total_client_upload_bytes > 0 else 0.0
                ),
            }
            if global_eval is not None:
                wb_metrics["global/val_loss"] = global_eval["loss"]
                wb_metrics["global/val_perplexity"] = global_eval["perplexity"]
                wb_metrics["global/val_tokens"] = global_eval["tokens"]
                wb_metrics["global/val_batches"] = global_eval["batches"]

            for client_id, loss in enumerate(client_losses):
                wb_metrics[f"client/{client_id}/loss"] = loss
                wb_metrics[f"client/{client_id}/perplexity"] = float(torch.exp(torch.tensor(loss)))
                wb_metrics[f"client/{client_id}/steps"] = executed_steps[client_id]
                wb_metrics[f"communication/client/{client_id}_to_server_mb"] = (
                    client_upload_bytes[client_id] / (1024 ** 2)
                )

            logger.log_round(wb_metrics, r)

            print(
                f"Round {r:03d} complete: mean_loss={mean_loss:.4f}, "
                f"client->server={total_client_upload_bytes / (1024 ** 2):.2f} MB, "
                f"time={elapsed:.1f}s"
            )

            if (r + 1) % cfg.save_every == 0:
                path = server.save(cfg.output_dir, r + 1)
                print(f"Saved: {path}")

            with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        final_path = server.save(cfg.output_dir, cfg.rounds)
        logger.log_final_model(final_path)
        logger.log_summary({
            "final_round": cfg.rounds - 1,
            "final_checkpoint": final_path,
            "total_tokens_seen": total_tokens,
            "num_clients": cfg.num_clients,
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
        })
        print(f"Final model logged to W&B as artifact from: {final_path}")
        print("Federated SLTrain finished.")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
