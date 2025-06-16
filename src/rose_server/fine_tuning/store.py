"""SQLModel-based storage for fine-tuning jobs with clean OpenAI compatibility."""
import logging
import uuid
from typing import Dict, List, Optional
from openai.types.fine_tuning import (
    FineTuningJob as OpenAIFineTuningJob,
)
from openai.types.fine_tuning import (
    FineTuningJobEvent as OpenAIFineTuningJobEvent,
)
from sqlalchemy import delete, func
from sqlmodel import select
from ..database import current_timestamp, run_in_session
from ..entities.fine_tuning import (
    FineTuningEvent,
    FineTuningJob,
)
logger = logging.getLogger(__name__)

class FineTuningStore:
    """SQLModel-based store for fine-tuning jobs."""

    async def create_job(
        self,
        model: str,
        training_file: str,
        hyperparameters: Optional[Dict] = None,
        suffix: Optional[str] = None,
        validation_file: Optional[str] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> OpenAIFineTuningJob:
        """Create a new fine-tuning job with normalized method storage."""
        job_id = f"ftjob-{uuid.uuid4().hex[:24]}"
        hp = hyperparameters or {}
        training_method = "supervised"
        method_config = {"type": training_method, training_method: {"hyperparameters": hp}}
        logger.info(f"Using method '{training_method}' with hyperparameters: {hp}")

        async def create_operation(session):
            job = FineTuningJob(
                id=job_id,
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
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job
        job = await run_in_session(create_operation)
        logger.info(f"Created fine-tuning job: {job_id} with method config stored as JSON")
        return job.to_openai()

    async def get_job(self, job_id: str) -> Optional[OpenAIFineTuningJob]:
        """Get a job by ID with pre-loaded method configuration."""

        async def get_operation(session):
            job = await session.get(FineTuningJob, job_id)
            return job.to_openai() if job else None
        return await run_in_session(get_operation, read_only=True)

    async def list_jobs(
        self, limit: int = 20, after: Optional[str] = None, metadata_filters: Optional[Dict] = None
    ) -> List[OpenAIFineTuningJob]:
        """List jobs with optional metadata filtering."""

        async def list_operation(session):
            statement = select(FineTuningJob).order_by(FineTuningJob.created_at.desc())
            if metadata_filters:
                for key, value in metadata_filters.items():
                    statement = statement.where(FineTuningJob.meta.op("JSON_EXTRACT")(f"$.{key}") == value)
            statement = statement.limit(limit)
            jobs = (await session.execute(statement)).scalars().all()
            return [job.to_openai() for job in jobs]
        return await run_in_session(list_operation, read_only=True)

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[Dict] = None,
        fine_tuned_model: Optional[str] = None,
        trained_tokens: Optional[int] = None,
    ) -> Optional[OpenAIFineTuningJob]:
        """Update job status."""

        async def update_operation(session):
            job = await session.get(FineTuningJob, job_id)
            if not job:
                return None
            job.status = status
            if error:
                job.error = error
            if fine_tuned_model:
                job.fine_tuned_model = fine_tuned_model
            if trained_tokens:
                job.trained_tokens = trained_tokens
            if status in ["succeeded", "failed", "cancelled"]:
                job.finished_at = current_timestamp()
            elif status == "running" and not job.started_at:
                job.started_at = current_timestamp()
            session.add(job)
            await session.commit()
            await session.refresh(job)
            logger.info(f"Updated job {job_id} status to {status}")
            return job.to_openai()
        return await run_in_session(update_operation)

    async def add_event(
        self,
        job_id: str,
        level: str,
        message: str,
        data: Optional[Dict] = None,
    ) -> str:
        """Add an event to a job."""
        event_id = f"ftevt-{uuid.uuid4().hex[:24]}"
        event = FineTuningEvent(
            id=event_id,
            job_id=job_id,
            created_at=current_timestamp(),
            level=level,
            message=message,
            data=data,
        )

        async def add_operation(session):
            session.add(event)
            await session.commit()
            return event_id
        result = await run_in_session(add_operation)
        logger.debug(f"Added event to job {job_id}: {message}")
        return result

    async def get_events(
        self,
        job_id: str,
        limit: int = 20,
        after: Optional[str] = None,
    ) -> List[OpenAIFineTuningJobEvent]:
        """Get events for a job."""

        async def get_events_operation(session):
            statement = (
                select(FineTuningEvent)
                .where(FineTuningEvent.job_id == job_id)
                .order_by(FineTuningEvent.created_at.asc())
            )
            events = (await session.execute(statement)).scalars().all()
            return [event.to_openai() for event in events]
        return await run_in_session(get_events_operation, read_only=True)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a fine-tuning job and all associated events."""

        async def delete_operation(session):

            event_count_result = await session.execute(
                select(func.count(FineTuningEvent.id)).where(FineTuningEvent.job_id == job_id)
            )
            event_count = event_count_result.scalar()
            await session.execute(delete(FineTuningEvent).where(FineTuningEvent.job_id == job_id))
            job = await session.get(FineTuningJob, job_id)
            if job:
                await session.delete(job)
                await session.commit()
                logger.info(f"Deleted job {job_id} and {event_count} associated events")
                return True
            return False
        return await run_in_session(delete_operation)

    async def update_job_result_files(self, job_id: str, result_files: List[str]) -> Optional[OpenAIFineTuningJob]:
        """Update job with result file IDs."""

        async def update_operation(session):
            job = await session.get(FineTuningJob, job_id)
            if not job:
                return None
            job.result_files = result_files
            session.add(job)
            await session.commit()
            await session.refresh(job)
            logger.info(f"Updated job {job_id} with result files: {result_files}")
            return job.to_openai()
        return await run_in_session(update_operation)

    async def mark_job_failed(self, job_id: str, error: str) -> Optional[OpenAIFineTuningJob]:
        """Mark a job as failed with error message."""

        async def update_operation(session):
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
                id=f"ftevt-{uuid.uuid4().hex[:24]}",
                job_id=job_id,
                created_at=current_timestamp(),
                level="error",
                message=f"Job failed: {error}",
                data={"error": error},
            )
            session.add(event)
            await session.commit()
            logger.info(f"Marked job {job_id} as failed: {error}")
            return job.to_openai()
        return await run_in_session(update_operation)

    async def get_stats(self) -> Dict:
        """Get store statistics."""

        async def stats_operation(session):
            job_count_result = await session.execute(select(func.count(FineTuningJob.id)))
            job_count = job_count_result.scalar()
            event_count_result = await session.execute(select(func.count(FineTuningEvent.id)))
            event_count = event_count_result.scalar()
            return {"total_jobs": job_count, "total_events": event_count, "storage_type": "sqlite"}
        return await run_in_session(stats_operation, read_only=True)