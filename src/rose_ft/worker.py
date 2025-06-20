"""Rose Fine-tuning Worker - processes fine-tuning jobs."""

import logging
import signal
import time
from typing import Any, Dict, Optional

import httpx
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from rose_core.config.service import HOST, LOG_FORMAT, LOG_LEVEL, MAX_CONCURRENT_TRAINING, PORT
from rose_core.models import cleanup_model_memory

from .training.trainer import run_training_job

logger = logging.getLogger(__name__)

# API configuration
BASE_URL = f"http://{HOST}:{PORT}"
POLL_INTERVAL = 5  # seconds
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
        # Update job status
        update_job_status(job_id, "running")

        # Run the training
        result = run_training_job(
            job_id=job_id,
            ft_job_id=ft_job_id,
            model_name=model_name,
            training_file=training_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
        )

        # Update job status based on result
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
        # Clean up memory
        cleanup_model_memory()


def update_job_status(job_id: int, status: str, result: Optional[Dict] = None) -> None:
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


def schedule_poller(scheduler: BackgroundScheduler) -> None:
    """Register a polling job that fetches new work every POLL_INTERVAL seconds."""

    def poll() -> None:
        try:
            # Check current load
            running_jobs = len([job for job in scheduler.get_jobs() if job.id.startswith("train_")])
            if running_jobs >= MAX_CONCURRENT_TRAINING:
                return

            # Poll for new jobs
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                response = client.get(
                    f"{BASE_URL}/v1/jobs",
                    params={
                        "type": "training",
                        "status": "queued",
                        "limit": MAX_CONCURRENT_TRAINING - running_jobs,
                    },
                )
                response.raise_for_status()

                data = response.json()
                jobs = data.get("data", [])

                for job in jobs:
                    job_id = str(job["id"])

                    # Skip if already scheduled
                    if scheduler.get_job(f"train_{job_id}"):
                        continue

                    logger.info(f"Found training job {job_id}")

                    # Schedule the job
                    scheduler.add_job(
                        process_training_job,
                        args=[int(job_id), job["payload"]],
                        id=f"train_{job_id}",
                        executor="training",
                    )
                    logger.info(f"Scheduled training job {job_id}")

        except Exception as exc:
            logger.error(f"Polling error: {exc}")

    # Schedule the poller to run every POLL_INTERVAL seconds
    scheduler.add_job(poll, "interval", seconds=POLL_INTERVAL, id="poller")

    # Run once immediately
    poll()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
    )

    logger.info(
        f"Rose Fine-tuning Worker starting - max concurrent: {MAX_CONCURRENT_TRAINING}, polling every {POLL_INTERVAL}s"
    )

    # Set up scheduler with thread pool
    executors = {
        "default": ThreadPoolExecutor(max_workers=1),
        "training": ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRAINING),
    }
    scheduler = BackgroundScheduler(executors=executors)
    scheduler.start()

    # Start polling for jobs
    schedule_poller(scheduler)

    # Set up signal handling
    stop_flag = {"quit": False}

    def handle_signal(*_: Any) -> None:
        stop_flag["quit"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Rose Fine-tuning Worker ready - polling every {POLL_INTERVAL}s")

    # Main loop
    try:
        while not stop_flag["quit"]:
            time.sleep(1)
    finally:
        logger.info("Shutting down...")
        scheduler.shutdown(wait=True)
        logger.info("Rose Fine-tuning Worker stopped")


if __name__ == "__main__":
    main()
