"""Fine-tuning type definitions."""

from typing import Any, Dict, Optional

from pydantic.dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Training hyperparameters."""

    # Core params (required)
    batch_size: int
    max_length: int
    n_epochs: int
    learning_rate: float

    # Training config (required)
    gradient_accumulation_steps: int
    validation_split: float
    early_stopping_patience: int
    warmup_ratio: float
    scheduler_type: str
    min_lr_ratio: float
    weight_decay: float

    # LoRA config (required)
    use_lora: bool

    # Other (required)
    seed: int
    suffix: str

    # Optional fields
    lora_config: Optional[Dict[str, Any]] = None
    fp16: Optional[bool] = None
    eval_batch_size: Optional[int] = None
