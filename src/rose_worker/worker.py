"""Dead simple worker - just run jobs."""

import logging
import os
import time

from .client import get_client
from .fine_tuning.process import process_training_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable PostHog analytics
os.environ["POSTHOG_DISABLED"] = "1"


def main() -> None:
    client = get_client()
    logger.info("Worker started")

    while True:
        try:
            # Get training jobs only
            jobs = client.get_queued_jobs("training", limit=1)
            if jobs:
                job = jobs[0]
                logger.info(f"Running training job {job['id']}")
                process_training_job(job["id"], job["payload"])
        except Exception as e:
            logger.error(f"Job failed: {e}")

        time.sleep(5)


if __name__ == "__main__":
    main()
