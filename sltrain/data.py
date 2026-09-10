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
