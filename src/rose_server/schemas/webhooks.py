"""Webhook-related schemas."""
from typing import Any, Dict
from pydantic import BaseModel

class WebhookEvent(BaseModel):
    """Webhook event payload."""

    event: str
    object: str
    job_id: int
    object_id: str
    created_at: int
    data: Dict[str, Any]