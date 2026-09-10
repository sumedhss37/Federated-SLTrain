from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .layers import replace_linear_layers
from .utils import get_torch_dtype


def build_model(cfg):
    config = AutoConfig.from_pretrained(cfg.model_name)
    # Pretraining from a fresh parameter initialization, while borrowing the
    # architecture/configuration and tokenizer from the model family.
    model = AutoModelForCausalLM.from_config(config)

    replaced = replace_linear_layers(
        model,
        rank=cfg.rank,
        sparsity=cfg.sparsity,
        alpha=cfg.lora_alpha,
        seed=cfg.seed,
        exclude_patterns=cfg.exclude_linear_patterns if not cfg.replace_lm_head else (),
    )

    dtype = get_torch_dtype(cfg.dtype)
    if dtype == torch.bfloat16:
        bf16_ok = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not bf16_ok:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = model.to(dtype=dtype)
    return model, replaced


def build_tokenizer(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
