from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class Config:
    # Model / data
    model_name: str = "EleutherAI/pythia-70m"
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    seed: int = 42
    num_clients: int = 2
    seq_len: int = 256
    batch_size: int = 2
    local_steps: int = 10
    rounds: int = 5

    # SLTrain
    rank: int = 16
    sparsity: float = 0.03
    lora_alpha: float = 16.0
    replace_lm_head: bool = False
    exclude_linear_patterns: tuple = ("lm_head",)

    # Local optimizer
    client_lr: float = 3e-4
    client_weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Federated outer optimizer
    server_aggregator: str = "fedavg"  # fedavg | optimizer
    server_lr: float = 1.0
    server_optimizer: str = "sgd"      # sgd | adamw
    server_weight_decay: float = 0.0

    # SparseLoCo-style communication
    compression_density: float = 0.02
    quant_bits: int = 2
    error_feedback_dtype: str = "float32"

    # Precision / runtime
    dtype: str = "bfloat16"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    log_every: int = 1
    save_every: int = 1
    output_dir: str = "/kaggle/working/fed_sltrain_outputs"

    # Streaming data
    allow_data_repeat: bool = False

    def to_dict(self):
        return asdict(self)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
