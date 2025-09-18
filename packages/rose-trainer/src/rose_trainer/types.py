"""Fine-tuning type definitions."""

from typing import List, Literal, Optional

from peft import TaskType
from pydantic import BaseModel


class LoraModelConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = []
    lora_dropout: float = 0.05
    bias: Literal["none"] = "none"
    task_type: str = TaskType.CAUSAL_LM


class Hyperparameters(BaseModel):
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
    lora_config: Optional[LoraModelConfig] = None
    fp16: Optional[bool] = None
    eval_batch_size: Optional[int] = None


class ModelConfig(BaseModel):
    id: str
    parent: Optional[str] = None
    model_name: str
    lora_target_modules: Optional[List[str]] = None
    quantization: Optional[str] = None
