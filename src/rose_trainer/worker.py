"""Training worker that processes fine-tuning jobs."""

import asyncio
import logging
import os

from .client import get_client
from .fine_tuning.process import process_training_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"


async def poll_training_jobs():
    """Poll for training jobs from the queue."""
    client = get_client()
    logger.info("Training worker started - polling for jobs")

    while True:
        try:
            jobs = client.get_queued_jobs("training", limit=1)
            if jobs:
                job = jobs[0]
                logger.info(f"Starting training job {job['id']}")
                # Run in thread to not block the event loop
                await asyncio.get_event_loop().run_in_executor(None, process_training_job, job["id"], job["payload"])
                logger.info(f"Completed training job {job['id']}")
        except Exception as e:
            logger.error(f"Training job failed: {e}")

        await asyncio.sleep(5)


async def run_trainer():
    """Run the training worker."""
    logger.info("Rose Trainer started")
    await poll_training_jobs()


def main():
    """Entry point for the trainer process."""
    asyncio.run(run_trainer())


if __name__ == "__main__":
    main()
