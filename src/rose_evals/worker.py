"""Rose Evals Worker - processes evaluation jobs."""

import logging
import os
import signal
import time
from typing import Any, Dict, Optional

import httpx
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from .evaluators.simple_evaluator import SimpleEvaluator

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("ROSE_BASE_URL", "http://localhost:8004")
POLL_INTERVAL = int(os.getenv("ROSE_EVAL_POLL_INTERVAL", "5"))
HTTP_TIMEOUT = 10
MAX_CONCURRENT_EVAL = int(os.getenv("ROSE_MAX_CONCURRENT_EVAL", "2"))


def post_webhook(event: str, job_id: int, eval_id: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Post webhook event to server."""
    payload = {
        "event": event,
        "object": "eval",
        "job_id": job_id,
        "object_id": eval_id,
        "created_at": int(time.time()),
        "data": data or {},
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            client.post(f"{BASE_URL}/v1/webhooks/jobs", json=payload)
    except Exception as exc:
        logger.warning(f"Webhook '{event}' failed: {exc}")


def process_eval_job(job_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single evaluation job."""
    eval_id = payload.get("eval_id")
    model = payload.get("model")
    metadata = payload.get("metadata", {})

    if not eval_id or not model:
        raise ValueError(f"Missing required fields: eval_id={eval_id}, model={model}")

    logger.info(f"Starting eval job {job_id} for eval {eval_id} with model {model}")

    # Update status to running
    post_webhook("job.running", job_id, eval_id)

    try:
        # Extract eval configuration
        data_source_config = metadata.get("eval_data_source_config", {})
        testing_criteria = metadata.get("eval_testing_criteria", [])
        run_data_source = metadata.get("data_source", {})
        eval_name = metadata.get("eval", metadata.get("run_name"))

        # Run evaluation
        evaluator = SimpleEvaluator()
        result = evaluator.run_evaluation(
            eval_run_id=eval_id,
            model=model,
            data_source_config=data_source_config,
            testing_criteria=testing_criteria,
            run_data_source=run_data_source,
            eval_name=eval_name,
        )

        # Post results
        post_webhook(
            "job.completed",
            job_id,
            eval_id,
            {"results": result.get("results", {}), "samples": result.get("samples", [])},
        )

        logger.info(f"Eval job {job_id} completed successfully")
        return {"eval_id": eval_id, "status": "completed", "results": result}

    except Exception as exc:
        logger.exception(f"Eval job {job_id} failed")
        post_webhook("job.failed", job_id, eval_id, {"error": {"message": str(exc), "code": "job_error"}})
        raise


def schedule_poller(scheduler: BackgroundScheduler) -> None:
    """Register a polling job that fetches new work every POLL_INTERVAL seconds."""

    def poll() -> None:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            try:
                # Get queued eval jobs
                response = client.get(f"{BASE_URL}/v1/jobs", params={"status": "queued", "type": "eval", "limit": 10})
                response.raise_for_status()

                jobs = response.json().get("data", [])

                for job in jobs:
                    job_id = str(job["id"])

                    # Skip if already scheduled
                    if scheduler.get_job(job_id):
                        continue

                    logger.info(f"Found eval job {job_id}")

                    # Get full job details
                    detail_response = client.get(f"{BASE_URL}/v1/jobs/{job_id}")
                    detail_response.raise_for_status()
                    job_detail = detail_response.json()

                    # Schedule the job
                    scheduler.add_job(
                        process_eval_job,
                        args=[int(job_id), job_detail["payload"]],
                        id=job_id,
                        executor="eval",
                    )
                    logger.info(f"Scheduled eval job {job_id}")

            except Exception as exc:
                logger.error(f"Polling error: {exc}")

    # Schedule the poller to run every POLL_INTERVAL seconds
    scheduler.add_job(poll, "interval", seconds=POLL_INTERVAL, id="poller")

    # Run once immediately
    poll()


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info(f"Rose Evals Worker starting - max concurrent: {MAX_CONCURRENT_EVAL}, polling every {POLL_INTERVAL}s")

    # Set up scheduler with thread pool
    executors = {
        "default": ThreadPoolExecutor(max_workers=1),
        "eval": ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EVAL),
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

    logger.info(f"Rose Evals Worker ready - polling every {POLL_INTERVAL}s")

    # Main loop
    try:
        while not stop_flag["quit"]:
            time.sleep(1)
    finally:
        logger.info("Shutting down...")
        scheduler.shutdown(wait=True)
        logger.info("Rose Evals Worker stopped")


if __name__ == "__main__":
    main()
