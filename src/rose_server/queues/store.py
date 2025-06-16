"""Job queue store for managing jobs in the database."""

import logging
from typing import Optional

from ..database import current_timestamp, run_in_session
from ..entities.jobs import Job

logger = logging.getLogger(__name__)


class JobStore:
    """Job queue store for managing jobs in the database."""

    def __init__(self):
        pass

    async def initialize(self):
        """Initialize the job store (no-op now, kept for compatibility)."""
        logger.info("JobStore initialized (database-only mode)")

    async def shutdown(self):
        """Shutdown the job store (no-op now, kept for compatibility)."""
        pass

    async def enqueue(self, job_type: str, payload: dict, max_attempts: int = 3) -> Job:
        """Add job to queue."""

        async def create_job_record(session):
            job = Job(
                type=job_type,
                status="queued",
                payload=payload,
                created_at=current_timestamp(),
                max_attempts=max_attempts,
                attempts=0,
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

        job = await run_in_session(create_job_record)
        logger.info(f"Job {job.id} of type {job_type} queued for processing by worker")
        return job

    async def complete_job(self, job_id: int, result: Optional[dict] = None) -> None:
        """Mark job as completed."""

        async def update_op(session):
            job = await session.get(Job, job_id)
            if job:
                job.status = "completed"
                job.completed_at = current_timestamp()
                job.result = result
                await session.commit()

        await run_in_session(update_op)

    async def fail_job(self, job_id: int, error: str) -> None:
        """Mark job as failed."""

        async def update_op(session):
            job = await session.get(Job, job_id)
            if job:
                job.attempts += 1
                if job.attempts < job.max_attempts:
                    job.status = "queued"
                    job.error = error
                else:
                    job.status = "failed"
                    job.completed_at = current_timestamp()
                    job.error = error
                await session.commit()

        await run_in_session(update_op)

    async def request_cancel(self, job_id: int) -> bool:
        """Request job cancellation."""

        async def update_op(session):
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

        return await run_in_session(update_op)

    async def check_cancellation(self, job_id: int) -> str:
        """Check job status (for worker to poll)."""

        async def check_op(session):
            job = await session.get(Job, job_id)
            return job.status if job else None

        return await run_in_session(check_op)

    async def mark_cancelled(self, job_id: int) -> None:
        """Mark job as cancelled (worker confirms)."""

        async def update_op(session):
            job = await session.get(Job, job_id)
            if job:
                job.status = "cancelled"
                job.completed_at = current_timestamp()
                await session.commit()

        await run_in_session(update_op)

    async def request_pause(self, job_id: int) -> bool:
        """Request job pause."""

        async def update_op(session):
            job = await session.get(Job, job_id)
            if job and job.status == "running":
                job.status = "pausing"
                await session.commit()
                return True
            return False

        return await run_in_session(update_op)

    async def mark_paused(self, job_id: int) -> None:
        """Mark job as paused (worker confirms)."""

        async def update_op(session):
            job = await session.get(Job, job_id)
            if job:
                job.status = "paused"
                await session.commit()

        await run_in_session(update_op)
