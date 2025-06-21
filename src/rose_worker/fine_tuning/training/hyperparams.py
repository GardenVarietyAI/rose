import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from rose_core.config.service import FINE_TUNING_DEFAULT_LEARNING_RATE, FINE_TUNING_DEFAULT_MAX_LENGTH

logger = logging.getLogger(__name__)


@dataclass
class ResolvedHyperParams:
    """Resolved hyperparameters with guaranteed types."""

    batch_size: int
    max_length: int
    n_epochs: int
    learning_rate_multiplier: float
    gradient_accumulation_steps: int
    validation_split: float
    early_stopping_patience: int
    warmup_ratio: float
    scheduler_type: str
    min_lr_ratio: float
    use_lora: bool
    lora_config: Dict[str, Any] | None
    seed: int
    suffix: str
    learning_rate: float


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
    lora_config: Dict[str, Any] | None = None
    seed: int = 42
    suffix: str = "custom"
    learning_rate: float = field(init=False)

    @classmethod
    def resolve(cls, raw: Dict[str, Any]) -> ResolvedHyperParams:
        """Coerce raw dict & 'auto' values into concrete numbers."""
        hp = cls(**raw)

        # Resolve all fields to guaranteed types
        batch_size = cls._auto_int(hp.batch_size, 1)
        max_length = cls._auto_int(hp.max_length, FINE_TUNING_DEFAULT_MAX_LENGTH)
        n_epochs = cls._auto_int(hp.n_epochs, 3)
        gradient_accumulation_steps = cls._auto_int(hp.gradient_accumulation_steps, 1)

        lr_mult = hp.learning_rate_multiplier
        if lr_mult is None or lr_mult == "auto":
            lr_mult = 1.0
        else:
            lr_mult = float(lr_mult)

        learning_rate = FINE_TUNING_DEFAULT_LEARNING_RATE * lr_mult

        return ResolvedHyperParams(
            batch_size=batch_size,
            max_length=max_length,
            n_epochs=n_epochs,
            learning_rate_multiplier=lr_mult,
            gradient_accumulation_steps=gradient_accumulation_steps,
            validation_split=hp.validation_split,
            early_stopping_patience=hp.early_stopping_patience,
            warmup_ratio=hp.warmup_ratio,
            scheduler_type=hp.scheduler_type,
            min_lr_ratio=hp.min_lr_ratio,
            use_lora=hp.use_lora,
            lora_config=hp.lora_config,
            seed=hp.seed,
            suffix=hp.suffix,
            learning_rate=learning_rate,
        )

    @staticmethod
    def _auto_int(val: int | str | None, fallback: int | None) -> int:
        if val is None or val == "auto":
            if fallback is None:
                return 1
            return int(fallback)
        return int(val)
