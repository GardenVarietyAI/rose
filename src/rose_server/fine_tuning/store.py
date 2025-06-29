import logging
from typing import Any, Dict

from sqlalchemy import func
from sqlmodel import select

from rose_server.database import get_session
from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob
from rose_server.entities.jobs import Job as QueueJob

logger = logging.getLogger(__name__)


async def get_job_status() -> Dict[str, Any]:
    async with get_session() as session:
        result = await session.execute(select(QueueJob.status, func.count(QueueJob.id)).group_by(QueueJob.status))
        status_counts = {row[0]: row[1] for row in result}
        result = await session.execute(select(QueueJob).order_by(QueueJob.created_at.desc()).limit(10))
        recent_jobs = [
            {
                "id": job.id,
                "type": job.type,
                "status": job.status,
                "attempts": job.attempts,
                "max_attempts": job.max_attempts,
                "created_at": job.created_at,
                "payload": job.payload,
            }
            for job in result.scalars().all()
        ]
        return {"status_counts": status_counts, "recent_jobs": recent_jobs}


async def get_stats() -> Dict[str, Any]:
    """Get store statistics."""

    async with get_session(read_only=True) as session:
        job_count_result = await session.execute(select(func.count(FineTuningJob.id)))
        job_count = job_count_result.scalar()
        event_count_result = await session.execute(select(func.count(FineTuningEvent.id)))
        event_count = event_count_result.scalar()
        return {"total_jobs": job_count, "total_events": event_count, "storage_type": "sqlite"}
