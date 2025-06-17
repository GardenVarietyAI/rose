import logging
from dataclasses import dataclass, field
from typing import Dict

from ...config import ServiceConfig

logger = logging.getLogger(__name__)


@dataclass
class HyperParams:
    """User-supplied or auto-resolved hyper-parameters."""

    batch_size: int | str | None = None
    max_length: int | str | None = None
    n_epochs: int | str | None = 3
    learning_rate_multiplier: float | str | None = 1.0
    gradient_accumulation_steps: int | str | None = None
    validation_split: float = 0.1
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1
    use_lora: bool = True
    lora_config: dict | None = None
    seed: int = 42
    suffix: str = "custom"
    learning_rate: float = field(init=False)

    @classmethod
    def resolve(cls, raw: Dict, optimised=None) -> "HyperParams":
        """Coerce raw dict & 'auto' values into concrete numbers."""
        hp = cls(**raw)
        default_batch_size = 1
        default_max_length = 2048
        default_grad_accum = 4
        if optimised:
            hp.batch_size = cls._auto_int(hp.batch_size, optimised.batch_size)
            hp.max_length = cls._auto_int(hp.max_length, optimised.max_length)
            hp.gradient_accumulation_steps = cls._auto_int(
                hp.gradient_accumulation_steps, optimised.gradient_accumulation_steps
            )
        else:
            hp.batch_size = cls._auto_int(hp.batch_size, default_batch_size)
            hp.max_length = cls._auto_int(hp.max_length, default_max_length)
            hp.gradient_accumulation_steps = cls._auto_int(hp.gradient_accumulation_steps, default_grad_accum)
        hp.n_epochs = cls._auto_int(hp.n_epochs, 3)
        lr_mult = hp.learning_rate_multiplier
        if lr_mult is None or lr_mult == "auto":
            lr_mult = 1.0
        else:
            lr_mult = float(lr_mult)
        hp.learning_rate_multiplier = lr_mult
        hp.learning_rate = ServiceConfig.FINE_TUNING_DEFAULT_LEARNING_RATE * lr_mult
        return hp

    @staticmethod
    def _auto_int(val: int | str | None, fallback: int | None) -> int:
        if val is None or val == "auto":
            if fallback is None:
                return 1
            return int(fallback)
        return int(val)
