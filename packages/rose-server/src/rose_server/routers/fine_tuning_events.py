import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request
from rose_server.entities.fine_tuning import FineTuningEvent
from rose_server.schemas.fine_tuning import FineTuningJobEventRequest, FineTuningJobEventResponse
from sqlalchemy import asc
from sqlmodel import select

router = APIRouter(prefix="/v1/fine_tuning/jobs/{job_id}/events")
logger = logging.getLogger(__name__)


@router.get("", response_model=dict)
async def list_fine_tuning_events(
    req: Request,
    job_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of events to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> Dict[str, Any]:
    statement = (
        select(FineTuningEvent).where(FineTuningEvent.job_id == job_id).order_by(asc(FineTuningEvent.created_at))  # type: ignore[arg-type]
    )
    async with req.app.state.get_db_session(read_only=True) as session:
        events = list((await session.execute(statement)).scalars().all())

    if not events:
        return {"object": "list", "data": [], "has_more": False}

    return {
        "object": "list",
        "data": [FineTuningJobEventResponse.from_entity(event) for event in events],
        "has_more": len(events) == limit,
    }


@router.post("")
async def add_job_event(req: Request, job_id: str, event: FineTuningJobEventRequest) -> Dict[str, Any]:
    event_entity = FineTuningEvent(
        job_id=job_id, created_at=int(time.time()), level=event.level, message=event.message, data=event.data
    )
    async with req.app.state.get_db_session() as session:
        session.add(event_entity)
        await session.commit()
        logger.debug(f"Added event to job {job_id}: {event_entity.message}")

    return {"status": "accepted"}
