"""ROSE Worker - handles both fine-tuning and evaluation jobs."""

import logging
import signal
import time
from typing import Any, Dict, Optional

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from rose_core.config.service import LOG_FORMAT, LOG_LEVEL, MAX_CONCURRENT_TRAINING
from rose_core.models import cleanup_model_memory

from .client import ServiceClient, update_job_status
from .evals.process import process_eval_job
from .fine_tuning.process import process_training_job

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5
MAX_CONCURRENT_EVAL = 2


class Worker:
    """Worker that handles both training and evaluation jobs."""

    def __init__(self) -> None:
        self.scheduler: Optional[BackgroundScheduler] = None
        self.stop_flag = {"quit": False}
        self.client = ServiceClient()

    def poll_jobs(self) -> None:
        """Poll for both training and evaluation jobs."""
        try:
            # Check current load for each job type
            if not self.scheduler:
                return
            running_training = len([job for job in self.scheduler.get_jobs() if job.id.startswith("train_")])
            running_evals = len([job for job in self.scheduler.get_jobs() if job.id.startswith("eval_")])

            # Poll for training jobs if we have capacity
            if running_training < MAX_CONCURRENT_TRAINING:
                self._poll_job_type("training", "train_", MAX_CONCURRENT_TRAINING - running_training)

            # Poll for eval jobs if we have capacity
            if running_evals < MAX_CONCURRENT_EVAL:
                self._poll_job_type("eval", "eval_", MAX_CONCURRENT_EVAL - running_evals)

        except Exception as exc:
            logger.error(f"Polling error: {exc}")

    def _poll_job_type(self, job_type: str, prefix: str, limit: int) -> None:
        """Poll for jobs of a specific type."""
        jobs = self.client.get_queued_jobs(job_type, limit)

        for job in jobs:
            job_id = str(job["id"])
            prefixed_id = f"{prefix}{job_id}"

            if self.scheduler and self.scheduler.get_job(prefixed_id):
                continue

            logger.info(f"Found {job_type} job {job_id}")

            # Get full job details for eval jobs
            if job_type == "eval":
                job_detail = self.client.get_job_details(job_id)
                if not job_detail:
                    continue
                payload = job_detail["payload"]
            else:
                payload = job["payload"]

            # Schedule the appropriate processor
            if self.scheduler:
                if job_type == "training":
                    self.scheduler.add_job(
                        self._wrap_training_job,
                        args=[int(job_id), payload],
                        id=prefixed_id,
                        executor="training",
                    )
                else:  # eval
                    self.scheduler.add_job(
                        self._wrap_eval_job,
                        args=[int(job_id), payload],
                        id=prefixed_id,
                        executor="eval",
                    )

            logger.info(f"Scheduled {job_type} job {job_id}")

    def _wrap_training_job(self, job_id: int, payload: Dict[str, Any]) -> None:
        """Wrapper to handle training job cleanup."""
        try:
            process_training_job(job_id, payload)
        finally:
            cleanup_model_memory()

    def _wrap_eval_job(self, job_id: int, payload: Dict[str, Any]) -> None:
        """Wrapper to handle eval job cleanup and status updates."""
        try:
            # Update job status to running
            update_job_status(job_id, "running")
            result = process_eval_job(job_id, payload)
            # Update job status to completed
            update_job_status(job_id, "completed", result)
        except Exception as e:
            logger.exception(f"Eval job {job_id} failed")
            update_job_status(job_id, "failed", {"error": str(e)})
        finally:
            cleanup_model_memory()

    def handle_signal(self, *_: Any) -> None:
        """Handle shutdown signals."""
        self.stop_flag["quit"] = True

    def run(self) -> None:
        """Main worker loop."""
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=LOG_FORMAT,
        )

        logger.info(
            f"ROSE Worker starting - "
            f"max training: {MAX_CONCURRENT_TRAINING}, "
            f"max eval: {MAX_CONCURRENT_EVAL}, "
            f"polling every {POLL_INTERVAL}s"
        )

        # Set up scheduler with thread pools
        executors = {
            "training": ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRAINING),
            "eval": ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EVAL),
        }
        self.scheduler = BackgroundScheduler(executors=executors)
        self.scheduler.start()

        # Schedule the poller
        self.scheduler.add_job(self.poll_jobs, "interval", seconds=POLL_INTERVAL, id="poller")
        self.poll_jobs()

        # Set up signal handling
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        logger.info("ROSE Worker ready")

        # Main loop
        try:
            while not self.stop_flag["quit"]:
                time.sleep(1)
        finally:
            logger.info("Shutting down...")
            self.scheduler.shutdown(wait=True)
            self.client.close()
            logger.info("ROSE Worker stopped")


def main() -> None:
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    main()
