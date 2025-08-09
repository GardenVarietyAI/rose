"""Webhook API router."""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from rose_server.models.deps import ModelRegistryDep
from rose_server.schemas.webhooks import WebhookEvent, WebhookResponse
from rose_server.webhooks.training import (
    handle_training_cancelled,
    handle_training_completed,
    handle_training_failed,
    handle_training_progress,
    handle_training_running,
)

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


async def process_webhook_event(event: WebhookEvent, registry: ModelRegistryDep) -> None:
    """Process webhook event asynchronously in background."""
    try:
        logger.info(f"Processing webhook: {event.event} for {event.object} job {event.job_id}")

        if event.object == "training":
            if event.event == "job.completed":
                await handle_training_completed(event, registry)
            elif event.event == "job.failed":
                await handle_training_failed(event)
            elif event.event == "job.progress":
                await handle_training_progress(event)
            elif event.event == "job.running":
                await handle_training_running(event)
            elif event.event == "job.cancelled":
                await handle_training_cancelled(event)
            else:
                logger.warning(f"Unknown training webhook event: {event.event}")
                return

            logger.info(f"Successfully processed webhook: {event.event} for job {event.job_id}")
        else:
            logger.error(f"Unknown webhook object type: {event.object}")

    except Exception as e:
        logger.error(f"Error processing webhook {event.event} for job {event.job_id}: {e}", exc_info=True)


@router.post("/jobs", response_model=WebhookResponse)
async def receive_job_webhook(
    event: WebhookEvent, background_tasks: BackgroundTasks, registry: ModelRegistryDep = None
) -> WebhookResponse:
    """Receive webhook notifications from worker processes."""
    logger.info(f"Received webhook: {event.event} for {event.object} job {event.job_id}")

    # Validate basic structure
    if event.object not in ["training"]:
        raise HTTPException(status_code=400, detail=f"Unknown webhook object type: {event.object}")

    # Queue processing in background and respond immediately
    background_tasks.add_task(process_webhook_event, event, registry)

    return WebhookResponse(status="accepted", message="Webhook queued for processing")
