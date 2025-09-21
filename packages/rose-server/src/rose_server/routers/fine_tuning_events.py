import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from rose_server.fine_tuning.events.store import add_event, get_events
from rose_server.schemas.fine_tuning import FineTuningJobEventRequest, FineTuningJobEventResponse

router = APIRouter(prefix="/v1/fine_tuning/jobs/{job_id}/events")
logger = logging.getLogger(__name__)


@router.get("", response_model=dict)
async def list_fine_tuning_events(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of events to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> Dict[str, Any]:
    """List events for a fine-tuning job."""
    events = await get_events(job_id, limit=limit, after=after)

    if not events:
        return {"object": "list", "data": [], "has_more": False}

    return {
        "object": "list",
        "data": [FineTuningJobEventResponse.from_entity(event) for event in events],
        "has_more": len(events) == limit,
    }


@router.post("")
async def add_job_event(job_id: str, event: FineTuningJobEventRequest) -> Dict[str, Any]:
    """Add an event to a fine-tuning job."""
    await add_event(
        job_id=job_id,
        level=event.level,
        message=event.message,
        data=event.data,
    )
    return {"status": "accepted"}
