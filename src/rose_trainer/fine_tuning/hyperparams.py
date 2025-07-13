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
        """Create HyperParams from raw dict.

        Note: All values should be provided by the API layer.
        This method expects a complete configuration.
        """
        return cls(
            batch_size=raw["batch_size"],
            max_length=raw["max_length"],
            n_epochs=raw["n_epochs"],
            learning_rate=raw["learning_rate"],
            gradient_accumulation_steps=raw["gradient_accumulation_steps"],
            validation_split=raw["validation_split"],
            early_stopping_patience=raw["early_stopping_patience"],
            warmup_ratio=raw["warmup_ratio"],
            scheduler_type=raw["scheduler_type"],
            min_lr_ratio=raw["min_lr_ratio"],
            use_lora=raw["use_lora"],
            lora_config=raw.get("lora_config"),  # Optional field
            seed=raw["seed"],
            suffix=raw["suffix"],
        )
