"""Webhook API router."""

import logging
from typing import Dict, Union

from fastapi import APIRouter, HTTPException

from rose_server.models.deps import ModelRegistryDep
from rose_server.schemas.webhooks import WebhookEvent
from rose_server.webhooks.training import (
    handle_training_cancelled,
    handle_training_completed,
    handle_training_failed,
    handle_training_progress,
    handle_training_running,
)

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


@router.post("/jobs")
async def receive_job_webhook(
    event: WebhookEvent, registry: ModelRegistryDep = None
) -> Union[str, None, Dict[str, str]]:
    """Receive webhook notifications from worker processes."""
    logger.info(f"Received webhook: {event.event} for {event.object} job {event.job_id}")
    try:
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
            return {"status": "ok", "message": "Training webhook processed"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown webhook object type: {event.object}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
