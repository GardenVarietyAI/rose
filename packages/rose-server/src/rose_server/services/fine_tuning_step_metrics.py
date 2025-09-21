import logging
from typing import Any, Dict, List, Optional

from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob

logger = logging.getLogger(__name__)


class StepMetrics:
    def __init__(
        self,
        step: int,
        train_loss: float,
        epoch: int,
        learning_rate: float,
        train_accuracy: Optional[float] = None,
        valid_loss: Optional[float] = None,
        valid_accuracy: Optional[float] = None,
    ) -> None:
        self.step = step
        self.train_loss = train_loss
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.train_accuracy = train_accuracy
        self.valid_loss = valid_loss
        self.valid_accuracy = valid_accuracy

    @classmethod
    def from_event(cls, data: Dict[str, Any]) -> "StepMetrics":
        return cls(
            step=int(data["step"]),
            train_loss=float(data["loss"]),
            epoch=int(data["epoch"]),
            learning_rate=float(data["learning_rate"]),
            train_accuracy=data.get("accuracy"),
            valid_loss=data.get("valid_loss"),
            valid_accuracy=data.get("valid_accuracy"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "train_loss": self.train_loss,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "train_accuracy": self.train_accuracy,
            "valid_loss": self.valid_loss,
            "valid_accuracy": self.valid_accuracy,
        }

    def has_validation_metrics(self) -> bool:
        return self.valid_loss is not None or self.valid_accuracy is not None


def build_training_results(
    job: FineTuningJob,
    events: List[FineTuningEvent],
    final_loss: float,
    steps: int,
    final_perplexity: Optional[float] = None,
    epochs_completed: Optional[float] = None,
) -> Dict[str, Any]:
    step_metrics: List[StepMetrics] = []

    for event in events:
        if isinstance(event.data, dict) and {"step", "loss", "epoch", "learning_rate"} <= event.data.keys():
            metric = StepMetrics.from_event(event.data)
            step_metrics.append(metric)

    epochs_from_steps = max((m.epoch for m in step_metrics)) if step_metrics else None

    training_time_seconds = None
    if job.started_at and job.finished_at:
        training_time_seconds = job.finished_at - job.started_at

    training_summary = {
        "final_loss": final_loss,
        "total_steps": steps,
        "epochs_completed": epochs_completed if epochs_completed is not None else epochs_from_steps,
        "training_time_seconds": training_time_seconds,
    }

    if final_perplexity is not None:
        training_summary["final_perplexity"] = final_perplexity

    existing_metrics = job.training_metrics or {}
    return {
        **existing_metrics,
        "object": "fine_tuning.job.training_results",
        "step_data": [m.to_dict() for m in step_metrics],
        "summary": training_summary,
        "hyperparameters_used": {
            "n_epochs": epochs_completed if epochs_completed is not None else epochs_from_steps,
            "batch_size": job.hyperparameters.get("batch_size") if job.hyperparameters else None,
            "learning_rate_multiplier": (
                job.hyperparameters.get("learning_rate_multiplier") if job.hyperparameters else None
            ),
        },
        "model_info": {
            "base_model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "training_file": job.training_file,
            "validation_file": job.validation_file,
            "trained_tokens": job.trained_tokens,
        },
    }
