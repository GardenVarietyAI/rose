import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rose_core.config.service import DATA_DIR
from rose_worker.client import ServiceClient, post_webhook

from . import HFTrainer

logger = logging.getLogger(__name__)

_client = ServiceClient()


def run_training_job(
    job_id: int,
    ft_job_id: str,
    model_name: str,
    training_file: str,
    hyperparameters: Dict[str, Any],
    suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single training job."""

    # Create event callback for progress reporting
    def event_callback(level: str, msg: str, data: Dict[str, Any] | None = None) -> None:
        if level in ["info", "warning", "error"]:
            post_webhook(
                "job.progress", "training", job_id, ft_job_id, {"level": level, "message": msg, **(data or {})}
            )

    # Create cancellation check callback
    def check_cancel_callback() -> str:
        return _client.check_fine_tuning_job_status(ft_job_id)

    # Send job running webhook
    post_webhook("job.running", "training", job_id, ft_job_id)

    # Add suffix to hyperparameters if provided
    if suffix:
        hyperparameters = hyperparameters.copy()
        hyperparameters["suffix"] = suffix

    try:
        # Initialize trainer
        trainer = HFTrainer()

        # Build training file path
        training_file_path = Path(DATA_DIR) / "uploads" / training_file

        # Run training
        result = trainer.train(
            job_id=ft_job_id,
            model_name=model_name,
            training_file_path=training_file_path,
            hyperparameters=hyperparameters,
            check_cancel_callback=check_cancel_callback,
            event_callback=event_callback,
        )

        if result.get("success"):
            # Send completion webhook
            data = {
                "fine_tuned_model": result["model_name"],
                "trained_tokens": int(result.get("tokens_processed", 0)),
                "final_loss": result.get("final_loss"),
                "steps": result.get("steps"),
            }
            post_webhook("job.completed", "training", job_id, ft_job_id, data)

            return {
                "success": True,
                "model_id": result["model_name"],
                "trained_tokens": data["trained_tokens"],
            }

        # Handle cancellation
        if result.get("cancelled"):
            post_webhook("job.cancelled", "training", job_id, ft_job_id)
            return {"cancelled": True}

        # Handle failure
        error_msg = result.get("error", "Training failed")
        post_webhook(
            "job.failed",
            "training",
            job_id,
            ft_job_id,
            {"error": {"message": error_msg, "code": "training_failed"}},
        )
        return {"success": False, "error": error_msg}

    except Exception as e:
        logger.exception("Training job failed")
        post_webhook(
            "job.failed",
            "training",
            job_id,
            ft_job_id,
            {"error": {"message": str(e), "code": "job_error"}},
        )
        return {"success": False, "error": str(e)}
