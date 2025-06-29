import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, func
from sqlmodel import select

from rose_server.database import current_timestamp, get_session
from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob

logger = logging.getLogger(__name__)


async def create_job(
    model: str,
    training_file: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    suffix: Optional[str] = None,
    validation_file: Optional[str] = None,
    seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FineTuningJob:
    """Create a new fine-tuning job with normalized method storage."""
    hp = hyperparameters or {}
    training_method = "supervised"
    method_config = {"type": training_method, training_method: {"hyperparameters": hp}}
    logger.info(f"Using method '{training_method}' with hyperparameters: {hp}")

    job = FineTuningJob(
        model=model,
        status="validating_files",
        training_file=training_file,
        validation_file=validation_file,
        created_at=current_timestamp(),
        seed=seed or 42,
        suffix=suffix,
        meta=metadata,
        hyperparameters=hp,
        method=method_config,
    )

    async with get_session() as session:
        session.add(job)
        await session.commit()
        await session.refresh(job)
        logger.info(f"Created fine-tuning job: {job.id} with method config stored as JSON")
        return job


async def get_job(job_id: str) -> Optional[FineTuningJob]:
    """Get a job by ID with pre-loaded method configuration."""
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)
        return job if job else None


async def list_jobs(
    limit: int = 20, after: Optional[str] = None, metadata_filters: Optional[Dict[str, Any]] = None
) -> List[FineTuningJob]:
    """List jobs with optional metadata filtering."""
    statement = select(FineTuningJob).order_by(FineTuningJob.created_at.desc())  # type: ignore[attr-defined]
    if metadata_filters:
        for key, value in metadata_filters.items():
            statement = statement.where(FineTuningJob.meta.op("JSON_EXTRACT")(f"$.{key}") == value)
    if after:
        statement = statement.where(FineTuningJob.id > after)
    statement = statement.limit(limit)

    async with get_session(read_only=True) as session:
        jobs = (await session.execute(statement)).scalars().all()
        return jobs


async def update_job_status(
    job_id: str,
    status: str,
    error: Optional[Dict[str, Any]] = None,
    fine_tuned_model: Optional[str] = None,
    trained_tokens: Optional[int] = None,
) -> Optional[FineTuningJob]:
    """Update job status."""

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
    """Delete a fine-tuning job and all associated events."""

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
    """Update job with result file IDs."""

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
    """Mark a job as failed with error message."""

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
