"""Evaluation job webhook handlers."""

import logging

from rose_server.database import run_in_session
from rose_server.entities.jobs import Job
from rose_server.evals.runs.store import update_eval_run_error, update_eval_run_results, update_eval_run_status
from rose_server.evals.samples.store import create_eval_sample
from rose_server.schemas.webhooks import WebhookEvent

logger = logging.getLogger(__name__)


async def handle_eval_webhook(event: WebhookEvent) -> dict:
    """Handle evaluation job webhooks."""
    if event.event == "job.completed":
        await _handle_eval_completed(event)
    elif event.event == "job.failed":
        await _handle_eval_failed(event)
    elif event.event == "job.running":
        await _handle_eval_running(event)
    else:
        logger.warning(f"Unknown eval webhook event: {event.event}")
    return {"status": "ok", "message": "Eval webhook processed"}


async def _handle_eval_completed(event: WebhookEvent) -> None:
    """Handle successful evaluation job completion."""
    await update_eval_run_results(
        event.object_id,
        results=event.data.get("results", {}),
    )
    await _persist_eval_samples(event)
    await _update_queue_job_completed(event)


async def _handle_eval_failed(event: WebhookEvent) -> None:
    """Handle failed evaluation job."""
    error_data = event.data.get("error", "Unknown error")
    # Extract message if error is a dict
    if isinstance(error_data, dict):
        error_msg = error_data.get("message", str(error_data))
    else:
        error_msg = str(error_data)
    await update_eval_run_error(event.object_id, error_msg)
    await _update_queue_job_failed(event, error_msg)


async def _persist_eval_samples(event: WebhookEvent) -> None:
    """Persist individual evaluation sample results."""
    samples = event.data.get("samples", [])
    if not samples:
        logger.info(f"No samples to persist for eval {event.object_id}")
        return
    failed_count = 0
    for sample in samples:
        try:
            await create_eval_sample(eval_run_id=event.object_id, **sample)
        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to persist sample {sample.get('sample_index')}: {e}")
    if failed_count > 0:
        logger.warning(f"Failed to persist {failed_count}/{len(samples)} samples for eval {event.object_id}")
    else:
        logger.info(f"Successfully persisted {len(samples)} samples for eval {event.object_id}")


async def _update_queue_job_completed(event: WebhookEvent) -> None:
    """Update queue job status to completed."""

    async def update_job(session):
        job = await session.get(Job, event.job_id)
        if job:
            job.status = "completed"
            job.completed_at = event.created_at
            job.result = event.data
            await session.commit()
        else:
            logger.warning(f"Queue job {event.job_id} not found")

    await run_in_session(update_job)


async def _update_queue_job_failed(event: WebhookEvent, error_msg: str) -> None:
    """Update queue job status to failed."""

    async def update_job(session):
        job = await session.get(Job, event.job_id)
        if job:
            job.status = "failed"
            job.completed_at = event.created_at
            job.error = error_msg
            await session.commit()
        else:
            logger.warning(f"Queue job {event.job_id} not found")

    await run_in_session(update_job)


async def _handle_eval_running(event: WebhookEvent) -> None:
    """Handle evaluation job starting to run."""
    await update_eval_run_status(event.object_id, status="running")

    async def update_job(session):
        job = await session.get(Job, event.job_id)
        if job:
            job.status = "running"
            job.started_at = event.created_at
            await session.commit()

    await run_in_session(update_job)
    logger.info(f"Eval job {event.object_id} is now running")
