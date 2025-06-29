import logging
import time
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select

from rose_server.database import current_timestamp, get_session
from rose_server.entities.jobs import Job
from rose_server.schemas.jobs import JobResponse

logger = logging.getLogger(__name__)


async def fetch_job(job_id: int) -> Union[Job, None]:
    async with get_session(read_only=True) as session:
        return await session.get(Job, job_id)


async def get_jobs(status: Optional[str] = None, type: Optional[str] = None, limit: int = 10) -> List[JobResponse]:
    async with get_session(read_only=True) as session:
        query = select(Job)
        if status:
            query = query.where(Job.status == status)
        if type:
            query = query.where(Job.type == type)
        query = query.limit(limit).order_by(Job.created_at.desc())
        result = await session.execute(query)
        jobs = result.scalars().all()
        return [
            JobResponse(
                id=job.id,
                type=job.type,
                status=job.status,
                payload=job.payload,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                error=job.error,
            )
            for job in jobs
        ]


async def enqueue(job_type: str, payload: Dict[str, Any], max_attempts: int = 3) -> Job:
    """Add job to queue."""
    job = Job(
        type=job_type,
        status="queued",
        payload=payload,
        created_at=current_timestamp(),
        max_attempts=max_attempts,
        attempts=0,
    )

    async with get_session() as session:
        session.add(job)
        await session.commit()
        await session.refresh(job)
        logger.info(f"Job {job.id} of type {job_type} queued for processing by worker")
        return job


async def complete_job(job_id: int, result: Optional[dict] = None) -> None:
    """Mark job as completed."""
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if job:
            job.status = "completed"
            job.completed_at = current_timestamp()
            job.result = result
            await session.commit()


async def fail_job(job_id: int, error: str) -> None:
    """Mark job as failed."""
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            return None

        job.attempts += 1
        if job.attempts < job.max_attempts:
            job.status = "queued"
            job.error = error
        else:
            job.status = "failed"
            job.completed_at = current_timestamp()
            job.error = error
        await session.commit()


async def request_cancel(job_id: int) -> bool:
    """Request job cancellation."""
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if job and job.status in ["queued", "running", "pausing"]:
            if job.status == "running":
                job.status = "cancelling"
            else:
                job.status = "cancelled"
                job.completed_at = current_timestamp()

            await session.commit()

            return True

        return False


async def check_cancellation(job_id: int) -> Union[str, None]:
    """Check job status (for worker to poll)."""
    async with get_session(read_only=True) as session:
        job = await session.get(Job, job_id)
        return job.status if job else None


async def mark_cancelled(job_id: int) -> None:
    """Mark job as cancelled (worker confirms)."""
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if job:
            job.status = "cancelled"
            job.completed_at = current_timestamp()
            await session.commit()


async def request_pause(job_id: int) -> bool:
    """Request job pause."""
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if job and job.status == "running":
            job.status = "pausing"
            await session.commit()
            return True
        return False


async def mark_paused(job_id: int) -> None:
    """Mark job as paused (worker confirms)."""
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if job:
            job.status = "paused"
            await session.commit()


async def update_job_status(job_id: int, status: str, result: Optional[Dict[str, Any]] = None) -> Union[Job, None]:
    async with get_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            return None

        job.status = status
        if status == "running" and not job.started_at:
            job.started_at = int(time.time())
        elif status in ["completed", "failed", "cancelled"] and not job.completed_at:
            job.completed_at = int(time.time())

        if result:
            job.result = result

        await session.commit()
        await session.refresh(job)
        return job
