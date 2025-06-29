import logging
from typing import Any, Dict, List, Optional

from openai.types.fine_tuning import FineTuningJobEvent as OpenAIFineTuningJobEvent
from sqlmodel import select

from rose_server.database import current_timestamp, get_session
from rose_server.entities.fine_tuning import FineTuningEvent

logger = logging.getLogger(__name__)


async def add_event(job_id: str, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Add an event to a job."""
    event = FineTuningEvent(
        job_id=job_id,
        created_at=current_timestamp(),
        level=level,
        message=message,
        data=data,
    )

    async with get_session() as session:
        session.add(event)
        await session.commit()
        await session.refresh(event)
        logger.debug(f"Added event to job {job_id}: {message}")
        return event.id


async def get_events(job_id: str, limit: int = 20, after: Optional[str] = None) -> List[OpenAIFineTuningJobEvent]:
    """Get events for a job."""
    statement = (
        select(FineTuningEvent).where(FineTuningEvent.job_id == job_id).order_by(FineTuningEvent.created_at.asc())
    )
    async with get_session(read_only=True) as session:
        events = (await session.execute(statement)).scalars().all()
        return [event.to_openai() for event in events]
