"""Training job processor."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rose_core.config.settings import settings
from rose_core.models import cleanup_model_memory
from rose_trainer.client import ServiceClient

from .training.hf_trainer import train

logger = logging.getLogger(__name__)


def process_training_job(job_id: int, payload: Dict[str, Any], client: ServiceClient) -> None:
    """Process a single training job."""
    ft_job_id: str = payload["job_id"]
    model_name: str = payload["model"]
    training_file: str = payload["training_file"]
    hyperparameters: Dict[str, Any] = payload.get("hyperparameters", {})
    suffix: Optional[str] = payload["suffix"]
    logger.info(f"Starting training job {job_id} for fine-tuning {ft_job_id}")

    # Create event callback for progress reporting
    def event_callback(level: str, msg: str, data: Dict[str, Any] | None = None) -> None:
        if level in ["info", "warning", "error"]:
            client.post_webhook(
                "job.progress", "training", job_id, ft_job_id, {"level": level, "message": msg, **(data or {})}
            )

    def check_cancel_callback() -> str:
        return client.check_fine_tuning_job_status(ft_job_id)

    try:
        # Send job running webhook
        client.post_webhook("job.running", "training", job_id, ft_job_id)

        # Add suffix to hyperparameters if provided
        if suffix:
            hyperparameters = hyperparameters.copy()
            hyperparameters["suffix"] = suffix

        training_file_path = Path(settings.data_dir) / "uploads" / training_file

        result = train(
            job_id=ft_job_id,
            model_name=model_name,
            training_file_path=training_file_path,
            hyperparameters=hyperparameters,
            client=client,
            check_cancel_callback=check_cancel_callback,
            event_callback=event_callback,
        )

        client.post_webhook(
            "job.completed",
            "training",
            job_id,
            ft_job_id,
            {
                "fine_tuned_model": result["model_name"],
                "trained_tokens": int(result.get("tokens_processed", 0)),
                "final_loss": result.get("final_loss"),
                "steps": result.get("steps"),
            },
        )

        # Handle cancellation
        if result.get("cancelled"):
            client.post_webhook("job.cancelled", "training", job_id, ft_job_id)

    except Exception as e:
        logger.exception(f"Training job {job_id} failed with unexpected exception")
        client.post_webhook(
            "job.failed",
            "training",
            job_id,
            ft_job_id,
            {"error": {"message": str(e), "code": "job_error"}},
        )
        raise
    finally:
        cleanup_model_memory()
