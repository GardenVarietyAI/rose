"""Training worker that processes fine-tuning jobs."""

import logging
import os
import time

from rose_trainer.client import get_client
from rose_trainer.fine_tuning.process import process_training_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"


def poll_training_jobs() -> None:
    """Poll for training jobs from the queue."""
    client = get_client()
    logger.info("Training worker started - polling for jobs")

    while True:
        try:
            jobs = client.get_queued_jobs("training", limit=1)
            if jobs:
                job = jobs[0]
                logger.info(f"Starting training job {job['id']}")
                # Process job synchronously
                process_training_job(job["id"], job["payload"])
                logger.info(f"Completed training job {job['id']}")
        except Exception as e:
            logger.error(f"Training job failed: {e}")

        time.sleep(5)


def main() -> None:
    """Entry point for the trainer process."""
    logger.info("Rose Trainer started")
    poll_training_jobs()


if __name__ == "__main__":
    main()
