"""OpenAI-compatible fine-tuning jobs API endpoints."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from rose_server.fine_tuning.events.store import get_events
from rose_server.schemas.fine_tuning import FineTuningJobEventResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/fine_tuning/jobs/{job_id}/events", response_model=dict)
async def list_fine_tuning_events(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of events to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> Dict[str, Any]:
    """List events for a fine-tuning job."""
    events = await get_events(job_id, limit=limit, after=after)

    if not events:
        raise HTTPException(status_code=404, detail=f"Events for fine-tuning job {job_id} not found")

    return {
        "object": "list",
        "data": [FineTuningJobEventResponse.from_entity(event) for event in events],
        "has_more": len(events) == limit,
    }
