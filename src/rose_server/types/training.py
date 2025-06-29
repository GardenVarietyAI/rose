"""Type definitions for training and fine-tuning modules."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None if conversion fails."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class StepMetrics:
    """Metrics for a single training step."""

    step: int
    train_loss: float
    epoch: int
    learning_rate: float
    train_accuracy: Optional[float] = None
    valid_loss: Optional[float] = None
    valid_accuracy: Optional[float] = None

    @classmethod
    def from_event(cls, data: Dict[str, Any]) -> "StepMetrics":
        """Create StepMetrics from event data."""
        return cls(
            step=int(data["step"]),
            train_loss=float(data["loss"]),
            epoch=int(data.get("epoch", 1)),
            learning_rate=float(data.get("learning_rate", 5e-6)),
            train_accuracy=_safe_float(data.get("accuracy")),
            valid_loss=_safe_float(data.get("valid_loss")),
            valid_accuracy=_safe_float(data.get("valid_accuracy")),
        )
