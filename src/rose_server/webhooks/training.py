"""Training job webhook handlers."""

import logging

from ..database import run_in_session
from ..entities.jobs import Job
from ..fine_tuning.store import FineTuningStore
from ..schemas.webhooks import WebhookEvent
from .results_output import create_result_file

logger = logging.getLogger(__name__)


async def handle_training_webhook(event: WebhookEvent) -> dict:
    """Handle training job webhooks."""
    if event.event == "job.completed":
        await _handle_training_completed(event)
    elif event.event == "job.failed":
        await _handle_training_failed(event)
    elif event.event == "job.progress":
        await _handle_training_progress(event)
    elif event.event == "job.running":
        await _handle_training_running(event)
    elif event.event == "job.cancelled":
        await _handle_training_cancelled(event)
    else:
        logger.warning(f"Unknown training webhook event: {event.event}")
    return {"status": "ok", "message": "Training webhook processed"}


async def _handle_training_completed(event: WebhookEvent) -> None:
    """Handle successful training job completion."""
    store = FineTuningStore()
    result_file_id = await _create_training_result_file(event)
    await store.update_job_status(
        event.object_id,
        status="succeeded",
        fine_tuned_model=event.data.get("fine_tuned_model"),
        trained_tokens=event.data.get("trained_tokens", 0),
    )
    if result_file_id:
        await store.update_job_result_files(event.object_id, [result_file_id])
    await _update_queue_job_completed(event)


async def _handle_training_failed(event: WebhookEvent) -> None:
    """Handle failed training job."""
    store = FineTuningStore()
    error_msg = event.data.get("error", {}).get("message", "Unknown error")
    await store.mark_job_failed(event.object_id, error_msg)
    await _update_queue_job_failed(event, error_msg)


async def _create_training_result_file(event: WebhookEvent) -> str | None:
    """Create result file for completed training job."""
    final_loss = event.data.get("final_loss")
    steps = event.data.get("steps")
    if final_loss is None or steps is None:
        logger.warning(f"Missing final_loss or steps for job {event.object_id}")
        return None
    try:
        result_file_id = await create_result_file(event.object_id, final_loss, steps)
        logger.info(f"Created result file {result_file_id} for job {event.object_id}")
        return result_file_id
    except Exception as e:
        logger.error(f"Failed to create result file: {e}")
        return None


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


async def _handle_training_progress(event: WebhookEvent) -> None:
    """Handle training progress events."""
    store = FineTuningStore()
    message = event.data.get("message", "Training progress")
    level = event.data.get("level", "info")
    await store.add_event(job_id=event.object_id, level=level, message=message, data=event.data)
    logger.debug(f"Added progress event for job {event.object_id}: {message}")


async def _handle_training_running(event: WebhookEvent) -> None:
    """Handle training job starting to run."""
    store = FineTuningStore()
    await store.update_job_status(event.object_id, status="running")

    async def update_job(session):
        job = await session.get(Job, event.job_id)
        if job:
            job.status = "running"
            job.started_at = event.created_at
            await session.commit()

    await run_in_session(update_job)
    logger.info(f"Training job {event.object_id} is now running")


async def _handle_training_cancelled(event: WebhookEvent) -> None:
    """Handle training job cancellation."""
    store = FineTuningStore()
    await store.update_job_status(event.object_id, status="cancelled")

    async def update_job(session):
        job = await session.get(Job, event.job_id)
        if job:
            job.status = "cancelled"
            job.completed_at = event.created_at
            await session.commit()

    await run_in_session(update_job)
    logger.info(f"Training job {event.object_id} was cancelled")
