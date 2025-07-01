from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class HyperParams:
    """Training hyperparameters."""

    batch_size: int = 1
    max_length: int = 512
    n_epochs: int = 3
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    validation_split: float = 0.1
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1
    use_lora: bool = True
    lora_config: Dict[str, Any] | None = None
    seed: int = 42
    suffix: str = "custom"

    @classmethod
    def resolve(cls, raw: Dict[str, Any]) -> "HyperParams":
        """Create HyperParams from raw dict."""
        return cls(
            batch_size=int(raw.get("batch_size", 1)),
            max_length=int(raw.get("max_length", 512)),
            n_epochs=int(raw.get("n_epochs", 3)),
            learning_rate=float(raw.get("learning_rate", 5e-5)),
            gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
            validation_split=float(raw.get("validation_split", 0.1)),
            early_stopping_patience=int(raw.get("early_stopping_patience", 3)),
            warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
            scheduler_type=raw.get("scheduler_type", "cosine"),
            min_lr_ratio=float(raw.get("min_lr_ratio", 0.1)),
            use_lora=bool(raw.get("use_lora", True)),
            lora_config=raw.get("lora_config"),
            seed=int(raw.get("seed", 42)),
            suffix=raw.get("suffix", "custom"),
        )
