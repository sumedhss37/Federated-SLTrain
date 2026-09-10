"""Streaming C4 client partitions for causal LM pretraining."""

from __future__ import annotations

from typing import Optional

import torch
from datasets import load_dataset


class C4ClientStream:
    def __init__(
        self,
        tokenizer,
        client_id: int,
        num_clients: int,
        seq_len: int,
        batch_size: int,
        dataset_name: str = "allenai/c4",
        dataset_config: str = "en",
        allow_repeat: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.allow_repeat = allow_repeat
        self.client_id = client_id
        self.num_clients = num_clients

        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        # Fixed non-overlapping client partition. This matches the federated goal
        # of client-local data while keeping all data access streaming-friendly.
        self.dataset = ds.shard(num_shards=num_clients, index=client_id)
        self.iterator = iter(self.dataset)
        self.token_buffer = []
        self.exhausted = False

    def _fill(self, target_tokens: int):
        while len(self.token_buffer) < target_tokens and not self.exhausted:
            try:
                row = next(self.iterator)
            except StopIteration:
                self.exhausted = True
                break
            text = row["text"]
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                self.token_buffer.extend(ids)

    def next_batch(self) -> Optional[dict]:
        needed = self.batch_size * self.seq_len
        self._fill(needed)
        if len(self.token_buffer) < needed:
            if self.allow_repeat:
                self.iterator = iter(self.dataset)
                self.exhausted = False
                self._fill(needed)
            else:
                return None

        if len(self.token_buffer) < needed:
            return None

        arr = self.token_buffer[:needed]
        del self.token_buffer[:needed]
        ids = torch.tensor(arr, dtype=torch.long).view(self.batch_size, self.seq_len)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


class C4ValidationSet:
    """Fixed C4 validation batches reused after every global aggregation.

    The batches are materialized once on CPU so every evaluated global model is
    scored on exactly the same tokens. This makes round-to-round validation
    loss/perplexity directly comparable.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        batch_size: int,
        num_batches: int = 16,
        dataset_name: str = "allenai/c4",
        dataset_config: str = "en",
    ):
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1")

        self.batches = []
        ds = load_dataset(
            dataset_name,
            dataset_config,
            split="validation",
            streaming=True,
        )
        iterator = iter(ds)
        token_buffer = []
        target = batch_size * seq_len

        while len(self.batches) < num_batches:
            while len(token_buffer) < target:
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                ids = tokenizer(
                    row["text"],
                    add_special_tokens=False,
                )["input_ids"]
                if ids:
                    token_buffer.extend(ids)

            if len(token_buffer) < target:
                break

            arr = token_buffer[:target]
            del token_buffer[:target]
            ids = torch.tensor(arr, dtype=torch.long).view(batch_size, seq_len)
            self.batches.append(
                {
                    "input_ids": ids,
                    "labels": ids.clone(),
                    "attention_mask": torch.ones_like(ids),
                }
            )

        if not self.batches:
            raise RuntimeError("Could not construct any C4 validation batches")

    def __len__(self):
        return len(self.batches)


@torch.no_grad()
def evaluate_model(model, validation_set: C4ValidationSet, device: torch.device):
    """Evaluate the global model on the fixed validation batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in validation_set.batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, use_cache=False)
        # HF causal-LM loss averages over valid next-token positions.
        # Weight by the number of predicted tokens so batch averaging remains
        # correct even if a future evaluator uses variable-sized batches.
        n_pred = batch["labels"].numel() - batch["labels"].shape[0]
        total_loss += float(outputs.loss.float().item()) * n_pred
        total_tokens += n_pred

    loss = total_loss / max(total_tokens, 1)
    perplexity = float(torch.exp(torch.tensor(loss)).item())
    return {
        "loss": loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
        "batches": len(validation_set),
    }
