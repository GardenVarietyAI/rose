"""OpenAI-compatible fine-tuning API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query
from openai.types.fine_tuning import FineTuningJob
from sqlmodel import func, select

from rose_core.config.service import ServiceConfig

from ..database import run_in_session
from ..entities.jobs import Job as QueueJob
from ..queues.facade import TrainingJob
from ..services import get_fine_tuning_store, get_job_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/v1/fine_tuning/queue/status")
async def get_queue_status() -> dict:
    """Get queue status for debugging."""

    async def get_status(session):
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

    return await run_in_session(get_status)


@router.post("/v1/fine_tuning/jobs", response_model=FineTuningJob)
async def create_fine_tuning_job(
    model: str = Body(..., description="The name of the model to fine-tune"),
    training_file: str = Body(..., description="The ID of the uploaded file for training"),
    hyperparameters: Optional[dict] = Body(default=None, description="Hyperparameters for training"),
    method: Optional[dict] = Body(default=None, description="Fine-tuning method configuration"),
    suffix: Optional[str] = Body(default=None, description="Suffix for the fine-tuned model"),
    validation_file: Optional[str] = Body(default=None, description="The ID of the uploaded file for validation"),
    seed: Optional[int] = Body(default=None, description="Random seed for reproducibility"),
    metadata: Optional[dict] = Body(default=None, description="Set of 16 key-value pairs for job metadata"),
) -> FineTuningJob:
    """Create a fine-tuning job."""
    try:
        store = get_fine_tuning_store()
        if method:
            method_type = method.get("type", "supervised")
            if method_type in method and isinstance(method[method_type], dict):
                method_config = method[method_type]
                if "hyperparameters" in method_config:
                    hyperparameters = method_config["hyperparameters"]
            if method_type not in ["supervised", "dpo"]:
                logger.warning(f"Unsupported fine-tuning method type: {method_type}, using supervised")
        if not hyperparameters:
            hyperparameters = {
                "n_epochs": ServiceConfig.FINE_TUNING_DEFAULT_EPOCHS,
                "batch_size": ServiceConfig.FINE_TUNING_DEFAULT_BATCH_SIZE,
                "learning_rate_multiplier": ServiceConfig.FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER,
            }
        job = await store.create_job(
            model=model,
            training_file=training_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
            validation_file=validation_file,
            seed=seed,
            metadata=metadata,
        )
        await store.update_job_status(job.id, "queued")
        await TrainingJob.dispatch(
            model=model, training_file=training_file, job_id=job.id, hyperparameters=hyperparameters, suffix=suffix
        )
        return job
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/fine_tuning/jobs/{job_id}", response_model=FineTuningJob)
async def retrieve_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Retrieve a fine-tuning job."""
    store = get_fine_tuning_store()
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return job


@router.get("/v1/fine_tuning/jobs", response_model=dict)
async def list_fine_tuning_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Number of jobs to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> dict:
    """List fine-tuning jobs."""
    store = get_fine_tuning_store()
    jobs = await store.list_jobs(limit=limit, after=after)
    return {"object": "list", "data": jobs, "has_more": len(jobs) == limit}


@router.post("/v1/fine_tuning/jobs/{job_id}/cancel", response_model=FineTuningJob)
async def cancel_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Cancel a fine-tuning job."""
    store = get_fine_tuning_store()
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    async def find_job(session):
        stmt = select(QueueJob).where(
            QueueJob.type == "training", func.json_extract(QueueJob.payload, "$.job_id") == job_id
        )
        result = await session.execute(stmt)
        row = result.first()
        return row[0] if row else None

    queue_job = await run_in_session(find_job)
    if queue_job:
        success = await get_job_store().request_cancel(queue_job.id)
        if not success:
            raise HTTPException(status_code=400, detail=f"Cannot cancel job {job_id}")
        if job.status in ["queued", "running"]:
            await store.update_job_status(job_id, "cancelling")
    else:
        if job.status in ["queued", "running"]:
            await store.update_job_status(job_id, "cancelled")
    updated_job = await store.get_job(job_id)
    return updated_job


@router.get("/v1/fine_tuning/jobs/{job_id}/events", response_model=dict)
async def list_fine_tuning_events(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of events to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> dict:
    """List events for a fine-tuning job."""
    store = get_fine_tuning_store()
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    events = await store.get_events(job_id, limit=limit, after=after)
    return {"object": "list", "data": events, "has_more": len(events) == limit}


@router.get("/v1/fine_tuning/jobs/{job_id}/checkpoints", response_model=dict)
async def list_fine_tuning_job_checkpoints(
    job_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Number of checkpoints to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> dict:
    """List checkpoints for a fine-tuning job."""
    store = get_fine_tuning_store()
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return {"object": "list", "data": [], "has_more": False}


@router.post("/v1/fine_tuning/jobs/{job_id}/pause", response_model=FineTuningJob)
async def pause_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Pause a fine-tuning job."""
    store = get_fine_tuning_store()
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    async def find_job(session):
        stmt = select(QueueJob).where(
            QueueJob.type == "training", func.json_extract(QueueJob.payload, "$.job_id") == job_id
        )
        result = await session.execute(stmt)
        row = result.first()
        return row[0] if row else None

    queue_job = await run_in_session(find_job)
    if queue_job:
        success = await get_job_store().request_pause(queue_job.id)
        if not success:
            raise HTTPException(status_code=400, detail=f"Cannot pause job {job_id}")
    else:
        raise HTTPException(status_code=400, detail=f"Job {job_id} not found in queue")
    updated_job = await store.get_job(job_id)
    return updated_job


@router.post("/v1/fine_tuning/jobs/{job_id}/resume", response_model=FineTuningJob)
async def resume_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Resume a paused fine-tuning job."""
    store = get_fine_tuning_store()
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    if job.status == "paused":
        await TrainingJob.dispatch(
            model=job.model,
            training_file=job.training_file,
            job_id=job.id,
            hyperparameters=job.hyperparameters if job.hyperparameters else {},
            suffix=job.suffix,
        )
        await store.update_job_status(job_id, "queued")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot resume job {job_id}, current status: {job.status}")
    updated_job = await store.get_job(job_id)
    return updated_job
