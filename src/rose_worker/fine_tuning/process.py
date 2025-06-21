"""Training job processor."""

import logging
from typing import Any, Dict, Optional

import httpx

from rose_core.config.service import HOST, PORT
from rose_core.models import cleanup_model_memory

from .training.trainer import run_training_job

logger = logging.getLogger(__name__)

BASE_URL = f"http://{HOST}:{PORT}"
HTTP_TIMEOUT = 30


def process_training_job(job_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single training job."""
    ft_job_id = payload["job_id"]
    model_name = payload["model"]
    training_file = payload["training_file"]
    hyperparameters = payload.get("hyperparameters", {})
    suffix = payload.get("suffix")

    logger.info(f"Starting training job {job_id} for fine-tuning {ft_job_id}")

    try:
        update_job_status(job_id, "running")
        result = run_training_job(
            job_id=job_id,
            ft_job_id=ft_job_id,
            model_name=model_name,
            training_file=training_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
        )

        if result.get("success"):
            update_job_status(job_id, "completed", result)
            logger.info(f"Training job {job_id} completed successfully")
            return {"job_id": job_id, "status": "completed", **result}
        elif result.get("cancelled"):
            update_job_status(job_id, "cancelled")
            logger.info(f"Training job {job_id} cancelled")
            return {"job_id": job_id, "status": "cancelled"}
        else:
            error = result.get("error", "Unknown error")
            update_job_status(job_id, "failed", {"error": error})
            logger.error(f"Training job {job_id} failed: {error}")
            return {"job_id": job_id, "status": "failed", "error": error}

    except Exception as e:
        logger.exception(f"Training job {job_id} failed with exception")
        update_job_status(job_id, "failed", {"error": str(e)})
        raise
    finally:
        cleanup_model_memory()


def update_job_status(job_id: int, status: str, result: Optional[Dict[str, Any]] = None) -> None:
    """Update job status in the API."""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.patch(
                f"{BASE_URL}/v1/jobs/{job_id}",
                json={"status": status, "result": result},
            )
            response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to update job {job_id} status: {e}")
