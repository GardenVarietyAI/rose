"""Webhook API router."""
import logging

from fastapi import APIRouter, HTTPException

from ..schemas.webhooks import WebhookEvent
from .evaluation import handle_eval_webhook
from .training import handle_training_webhook

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)
@router.post("/jobs")

async def receive_job_webhook(event: WebhookEvent):
    """Receive webhook notifications from worker processes."""
    logger.info(f"Received webhook: {event.event} for {event.object} job {event.job_id}")
    try:
        if event.object == "training":
            return await handle_training_webhook(event)
        elif event.object == "eval":
            return await handle_eval_webhook(event)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown webhook object type: {event.object}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))