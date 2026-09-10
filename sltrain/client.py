from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
from tqdm.auto import tqdm

from .compression import compress_state_dict
from .client_optim import build_client_optimizer
from .data import C4ClientStream
from .utils import load_trainable_state, optimizer_state_to_cpu, trainable_state_dict


@dataclass
class ClientPersistentState:
    error_residuals: Dict[str, torch.Tensor] = field(default_factory=dict)
    optimizer_state: dict | None = None
    steps_seen: int = 0


class FederatedClient:
    """One client executed sequentially on a single Kaggle GPU.

    Only one client model is kept on GPU. Optimizer states and error-feedback
    residuals are persisted on CPU between client updates/rounds.
    """

    def __init__(self, client_id, cfg, model, tokenizer):
        self.client_id = client_id
        self.cfg = cfg
        self.model = model
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.stream = C4ClientStream(
            tokenizer=tokenizer,
            client_id=client_id,
            num_clients=cfg.num_clients,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            allow_repeat=cfg.allow_data_repeat,
        )

        self.optimizer = build_client_optimizer(self.model, cfg)
        self.state = ClientPersistentState()

    def _restore_optimizer(self):
        if self.state.optimizer_state is not None:
            self.optimizer.load_state_dict(self.state.optimizer_state)

    def _save_optimizer(self):
        self.state.optimizer_state = optimizer_state_to_cpu(self.optimizer.state_dict())

    def train_round(self, global_state: Dict[str, torch.Tensor]):
        load_trainable_state(self.model, global_state, self.device)
        self._restore_optimizer()
        self.model.train()

        total_loss = 0.0
        executed = 0
        iterator = range(self.cfg.local_steps)
        for _ in iterator:
            batch = self.stream.next_batch()
            if batch is None:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            loss.backward()
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            executed += 1
            self.state.steps_seen += 1

        self._save_optimizer()

        # delta = theta_start - theta_end, so the result is gradient-like.
        end_state = trainable_state_dict(self.model)
        delta = {name: global_state[name].float() - end_state[name].float() for name in global_state}

        if self.cfg.compression_mode.lower() == "none":
            payload = {name: value.clone() for name, value in delta.items()}
        elif self.cfg.compression_mode.lower() == "sparse":
            payload = compress_state_dict(
                delta,
                self.state.error_residuals,
                density=self.cfg.compression_density,
            )
        else:
            raise ValueError(f"Unknown compression_mode={self.cfg.compression_mode}")

        # Explicitly free transient GPU memory before the next client.
        del end_state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        tokens_seen = executed * self.cfg.batch_size * self.cfg.seq_len
        return payload, (total_loss / max(executed, 1), executed, tokens_seen)
