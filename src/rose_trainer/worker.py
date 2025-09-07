"""Training worker that processes fine-tuning jobs."""

import logging
import os

from rose_trainer.client import ServiceClient
from rose_trainer.process import process_training_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"


def process_next_training_job() -> bool:
    """Process the next training job in the queue.

    Returns:
        True if a job was processed, False if queue was empty
    """
    client = ServiceClient()

    try:
        jobs = client.get_queued_jobs("training", limit=1)
        if not jobs:
            logger.debug("No training jobs in queue")
            return False

        job = jobs[0]
        job_id = job["id"]  # Now string instead of int
        logger.info(f"Starting training job {job_id}")

        # Process job synchronously
        process_training_job(job_id, job["payload"], client)

        logger.info(f"Completed training job {job_id}")
        return True

    except Exception as e:
        logger.error(f"Training job failed: {e}")
        # Return False to indicate job processing failed
        # This allows the scheduler to handle retries appropriately
        return False
    finally:
        # Ensure client is cleaned up
        client.close()


def main() -> None:
    """Entry point for direct execution (useful for testing)."""
    logger.info("Rose Trainer - processing single job")
    processed = process_next_training_job()
    if processed:
        logger.info("Job processed successfully")
    else:
        logger.info("No jobs to process")


if __name__ == "__main__":
    main()
