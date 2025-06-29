"""OpenAI-compatible fine-tuning jobs API endpoints."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from openai.types.fine_tuning import FineTuningJob

from rose_core.config.service import (
    FINE_TUNING_DEFAULT_BATCH_SIZE,
    FINE_TUNING_DEFAULT_EPOCHS,
    FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER,
)
from rose_server.fine_tuning.events.store import get_events
from rose_server.queues.store import enqueue, find_job_by_payload_field, request_cancel, request_pause

from .store import create_job, get_job, list_jobs, update_job_status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/fine_tuning/jobs", response_model=FineTuningJob)
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
                "n_epochs": FINE_TUNING_DEFAULT_EPOCHS,
                "batch_size": FINE_TUNING_DEFAULT_BATCH_SIZE,
                "learning_rate_multiplier": FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER,
            }

        job = await create_job(
            model=model,
            training_file=training_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
            validation_file=validation_file,
            seed=seed,
            metadata=metadata,
        )

        await update_job_status(job.id, "queued")

        await enqueue(
            job_type="training",
            payload={
                "model": model,
                "training_file": training_file,
                "job_id": job.id,
                "hyperparameters": hyperparameters,
                "suffix": suffix or "custom",
            },
            max_attempts=3,
        )

        return job.to_openai()
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fine_tuning/jobs", response_model=dict)
async def list_fine_tuning_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Number of jobs to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> Dict[str, Any]:
    """List fine-tuning jobs."""
    jobs = await list_jobs(limit=limit, after=after)
    return {
        "object": "list",
        "data": [job.to_openai() for job in jobs],
        "has_more": len(jobs) == limit,
    }


@router.get("/v1/fine_tuning/jobs/{job_id}", response_model=FineTuningJob)
async def retrieve_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Retrieve a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return job.to_openai()


@router.post("/v1/fine_tuning/jobs/{job_id}/cancel", response_model=FineTuningJob)
async def cancel_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Cancel a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    queue_job = await find_job_by_payload_field("training", "job_id", job_id)
    if queue_job:
        success = await request_cancel(queue_job.id)
        if not success:
            raise HTTPException(status_code=400, detail=f"Cannot cancel job {job_id}")
        if job.status in ["queued", "running"]:
            await update_job_status(job_id, "cancelling")
    else:
        if job.status in ["queued", "running"]:
            await update_job_status(job_id, "cancelled")
    updated_job = await get_job(job_id)

    return updated_job.to_openai()


@router.get("/v1/fine_tuning/jobs/{job_id}/events", response_model=dict)
async def list_fine_tuning_events(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of events to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> Dict[str, Any]:
    """List events for a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    events = await get_events(job_id, limit=limit, after=after)
    return {"object": "list", "data": [event.to_openai() for event in events], "has_more": len(events) == limit}


@router.get("/v1/fine_tuning/jobs/{job_id}/checkpoints", response_model=dict)
async def list_fine_tuning_job_checkpoints(job_id: str) -> Dict[str, Any]:
    """List checkpoints for a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return {"object": "list", "data": [], "has_more": False}


@router.post("/v1/fine_tuning/jobs/{job_id}/pause", response_model=FineTuningJob)
async def pause_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Pause a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    queue_job = await find_job_by_payload_field("training", "job_id", job_id)
    if queue_job:
        success = await request_pause(queue_job.id)
        if not success:
            raise HTTPException(status_code=400, detail=f"Cannot pause job {job_id}")
    else:
        raise HTTPException(status_code=400, detail=f"Job {job_id} not found in queue")

    updated_job = await get_job(job_id)

    return updated_job.to_openai()


@router.post("/v1/fine_tuning/jobs/{job_id}/resume", response_model=FineTuningJob)
async def resume_fine_tuning_job(job_id: str) -> FineTuningJob:
    """Resume a paused fine-tuning job."""

    job = await get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Fine-tuning job {job_id} not found",
        )

    if job.status == "queued":
        await enqueue(
            job_type="training",
            payload={
                "model": job.model,
                "training_file": job.training_file,
                "job_id": job.id,
                "hyperparameters": job.hyperparameters if job.hyperparameters else {},
                "suffix": job.suffix or "custom",
            },
            max_attempts=3,
        )
        await update_job_status(job_id, "queued")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot resume job {job_id}, current status: {job.status}")

    updated_job = await get_job(job_id)

    return updated_job.to_openai()
