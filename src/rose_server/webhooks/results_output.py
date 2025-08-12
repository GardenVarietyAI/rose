import json
import logging
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

from rose_server.files.store import create_file
from rose_server.fine_tuning.events.store import get_events
from rose_server.fine_tuning.jobs.store import get_job, update_job_result_files

logger = logging.getLogger(__name__)


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


async def create_result_file(
    job_id: str, final_loss: float, steps: int, final_perplexity: Optional[float] = None
) -> Optional[str]:
    """
    Build a training-results artifact and upload it to the file store.
    Returns the file ID or *None* on failure.
    """

    job = await get_job(job_id)
    if job is None:
        logger.error("Job %s not found - aborting result-file creation", job_id)
        return None

    step_metrics: List[StepMetrics] = []
    train_loss_values: List[float] = []
    training_start: Optional[int] = None
    training_end: Optional[int] = None

    events = await get_events(job_id, limit=1000)
    for event in events:
        if "Training started" in event.message:
            training_start = event.created_at
        elif "Training completed" in event.message:
            training_end = event.created_at
        if isinstance(event.data, dict) and {"step", "loss"} <= event.data.keys():
            metric = StepMetrics.from_event(event.data)
            step_metrics.append(metric)
            train_loss_values.append(metric.train_loss)

    epochs_completed = max((m.epoch for m in step_metrics), default=1)

    training_summary = {
        "final_loss": final_loss,
        "final_train_loss": train_loss_values[-1] if train_loss_values else final_loss,
        "total_steps": steps,
        "epochs_completed": epochs_completed,
        "best_train_loss": min(train_loss_values, default=final_loss),
        "loss_improvement": (train_loss_values[0] - train_loss_values[-1] if len(train_loss_values) >= 2 else 0.0),
        "convergence_achieved": final_loss < 1.0,
        "training_time_seconds": (training_end - training_start) if training_start and training_end else None,
    }

    # Add perplexity if available (validation split was used)
    if final_perplexity is not None:
        training_summary["final_perplexity"] = final_perplexity

    training_results = {
        "object": "fine_tuning.job.training_results",
        "data": [asdict(m) for m in step_metrics],
        "summary": training_summary,
        "hyperparameters_used": {
            "n_epochs": epochs_completed,
            "batch_size": getattr(getattr(job, "hyperparameters", None), "batch_size", "auto"),
            "learning_rate_multiplier": getattr(
                getattr(job, "hyperparameters", None), "learning_rate_multiplier", "auto"
            ),
        },
        "model_info": {
            "base_model": job.model,
            "fine_tuned_model": job.fine_tuned_model,
            "training_file": job.training_file,
            "validation_file": job.validation_file,
            "trained_tokens": getattr(job, "trained_tokens", None),
        },
    }

    try:
        payload = json.dumps(training_results, indent=2, ensure_ascii=False)
        result_bytes = BytesIO(payload.encode())
        filename = f"ft-{job_id}-training-results.jsonl"
        file_obj = await create_file(
            result_bytes.getbuffer().nbytes, "fine-tune-results", filename, result_bytes.getvalue()
        )
        await update_job_result_files(job_id, [file_obj.id])
        logger.info("Result-file %s created for job %s (%d steps)", file_obj.id, job_id, len(step_metrics))
        file_obj_id: str = file_obj.id
        return file_obj_id
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to create result file for job {job_id}: {str(e)}", exc_info=True)
        return None
