"""Dead simple worker - just run jobs."""

import logging
import os
import time

from .client import get_client
from .evals.process import process_eval_job
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
            # Get any job
            for job_type in ["training", "eval"]:
                jobs = client.get_queued_jobs(job_type, limit=1)
                if jobs:
                    job = jobs[0]
                    logger.info(f"Running {job_type} job {job['id']}")

                    if job_type == "training":
                        process_training_job(job["id"], job["payload"])
                    else:
                        process_eval_job(job["id"], job["payload"])

                    break
        except Exception as e:
            logger.error(f"Job failed: {e}")

        time.sleep(5)


if __name__ == "__main__":
    main()
