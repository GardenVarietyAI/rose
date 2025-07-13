import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func
from sqlmodel import select

from rose_server.database import current_timestamp, get_session
from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob

logger = logging.getLogger(__name__)


async def create_job(ft_job: FineTuningJob) -> FineTuningJob:
    async with get_session() as session:
        session.add(ft_job)
        await session.commit()
        await session.refresh(ft_job)
        logger.info(f"Created fine-tuning job: {ft_job.id} with method config stored as JSON")
        return ft_job


async def get_job(job_id: str) -> Optional[FineTuningJob]:
    """Get a job by ID with pre-loaded method configuration."""
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)
        return job if job else None


async def list_jobs(
    limit: int = 20, after: Optional[str] = None, metadata_filters: Optional[Dict[str, Any]] = None
) -> List[FineTuningJob]:
    """List jobs with optional metadata filtering."""
    statement = select(FineTuningJob).order_by(FineTuningJob.created_at.desc())
    if metadata_filters:
        for key, value in metadata_filters.items():
            statement = statement.where(FineTuningJob.meta.op("JSON_EXTRACT")(f"$.{key}") == value)
    if after:
        statement = statement.where(FineTuningJob.id > after)
    statement = statement.limit(limit)

    async with get_session(read_only=True) as session:
        jobs = (await session.execute(statement)).scalars().all()
        return list(jobs)


async def update_job_status(
    job_id: str,
    status: str,
    error: Optional[Dict[str, Any]] = None,
    fine_tuned_model: Optional[str] = None,
    trained_tokens: Optional[int] = None,
    training_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[FineTuningJob]:
    async with get_session() as session:
        job = await session.get(FineTuningJob, job_id)
        if not job:
            return None

        if fine_tuned_model:
            job.fine_tuned_model = fine_tuned_model

        if trained_tokens:
            job.trained_tokens = trained_tokens

        if error:
            job.error = error

        if training_metrics:
            job.training_metrics = training_metrics

        job.status = status
        if status in ["succeeded", "failed", "cancelled"]:
            job.finished_at = current_timestamp()
        elif status == "running" and not job.started_at:
            job.started_at = current_timestamp()

        session.add(job)
        await session.commit()
        await session.refresh(job)
        logger.info(f"Updated job {job_id} status to {status}")
        return job


async def delete_job(job_id: str) -> bool:
    async with get_session() as session:
        event_count_result = await session.execute(
            select(func.count(FineTuningEvent.id)).where(FineTuningEvent.job_id == job_id)
        )
        event_count = event_count_result.scalar()

        await session.execute(delete(FineTuningEvent).where(FineTuningEvent.job_id == job_id))

        job = await session.get(FineTuningJob, job_id)
        if not job:
            return False

        await session.delete(job)
        await session.commit()
        logger.info(f"Deleted job {job_id} and {event_count} associated events")
        return True


async def update_job_result_files(job_id: str, result_files: List[str]) -> Optional[FineTuningJob]:
    async with get_session() as session:
        job = await session.get(FineTuningJob, job_id)
        if not job:
            return None
        job.result_files = result_files
        session.add(job)
        await session.commit()
        await session.refresh(job)
        logger.info(f"Updated job {job_id} with result files: {result_files}")
        return job


async def mark_job_failed(job_id: str, error: str) -> Optional[FineTuningJob]:
    async with get_session() as session:
        job = await session.get(FineTuningJob, job_id)
        if not job:
            return None
        job.status = "failed"
        job.finished_at = current_timestamp()
        job.error = {"message": error, "code": "job_error"}
        session.add(job)
        await session.commit()
        await session.refresh(job)
        event = FineTuningEvent(
            job_id=job_id,
            created_at=current_timestamp(),
            level="error",
            message=f"Job failed: {error}",
            data={"error": error},
        )
        session.add(event)
        await session.commit()
        logger.info(f"Marked job {job_id} as failed: {error}")
        return job
