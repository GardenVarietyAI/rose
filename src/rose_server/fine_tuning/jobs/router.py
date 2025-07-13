import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from rose_server.config.settings import settings
from rose_server.fine_tuning.jobs.store import create_job, get_job, list_jobs, update_job_status
from rose_server.queues.store import enqueue, find_job_by_payload_field, request_cancel, request_pause
from rose_server.schemas.fine_tuning import FineTuningJobCreateRequest, FineTuningJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Default hyperparameters for fine-tuning
FINE_TUNING_DEFAULT_EPOCHS = 3
FINE_TUNING_DEFAULT_BATCH_SIZE = "auto"
FINE_TUNING_DEFAULT_LEARNING_RATE_MULTIPLIER = "auto"

# Base learning rate for OpenAI-compatible multiplier.
# The value 2e-5 is a commonly used default for fine-tuning transformer models, 
# as it often provides a good balance between convergence speed and stability.
# This value can be overridden via the settings module if needed.
BASE_LEARNING_RATE = settings.get("BASE_LEARNING_RATE", 2e-5)


@router.post("/fine_tuning/jobs", response_model=FineTuningJobResponse)
async def create_fine_tuning_job(request: FineTuningJobCreateRequest) -> FineTuningJobResponse:
    """Create a fine-tuning job."""
    try:
        hyperparameters = request.hyperparameters

        if request.method:
            method_type = request.method.get("type", "supervised")
            if method_type in request.method and isinstance(request.method[method_type], dict):
                method_config = request.method[method_type]
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

        # Normalize hyperparameters before storing
        if "learning_rate_multiplier" in hyperparameters:
            multiplier = hyperparameters["learning_rate_multiplier"]
            hyperparameters["base_learning_rate"] = BASE_LEARNING_RATE

            if multiplier == "auto":
                # For auto, we use 1.0 as the multiplier
                hyperparameters["learning_rate_multiplier"] = 1.0
                hyperparameters["learning_rate"] = BASE_LEARNING_RATE
            else:
                hyperparameters["learning_rate_multiplier"] = float(multiplier)
                hyperparameters["learning_rate"] = BASE_LEARNING_RATE * float(multiplier)

        # Resolve "auto" batch_size
        if hyperparameters.get("batch_size") == "auto":
            hyperparameters["batch_size"] = 4

        # Resolve "auto" n_epochs
        if hyperparameters.get("n_epochs") == "auto":
            hyperparameters["n_epochs"] = 3

        job = await create_job(
            model=request.model,
            training_file=request.training_file,
            hyperparameters=hyperparameters,
            suffix=request.suffix,
            validation_file=request.validation_file,
            seed=request.seed,
            metadata=request.metadata,
        )

        await enqueue(
            job_type="training",
            payload={
                "model": request.model,
                "training_file": request.training_file,
                "job_id": job.id,
                "hyperparameters": hyperparameters,
                "suffix": request.suffix or "custom",
                "config": {
                    "data_dir": settings.data_dir,
                    "checkpoint_dir": settings.fine_tuning_checkpoint_dir,
                    "checkpoint_interval": settings.fine_tuning_checkpoint_interval,
                    "max_checkpoints": settings.fine_tuning_max_checkpoints,
                    "webhook_url": settings.webhook_url,
                },
            },
            max_attempts=3,
        )

        return FineTuningJobResponse.from_entity(job)
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
        "data": [FineTuningJobResponse.from_entity(job) for job in jobs],
        "has_more": len(jobs) == limit,
    }


@router.get("/fine_tuning/jobs/{job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    """Retrieve a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return FineTuningJobResponse.from_entity(job)


@router.post("/fine_tuning/jobs/{job_id}/cancel", response_model=FineTuningJobResponse)
async def cancel_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
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

    return FineTuningJobResponse.from_entity(updated_job)


@router.get("/fine_tuning/jobs/{job_id}/checkpoints", response_model=dict)
async def list_fine_tuning_job_checkpoints(job_id: str) -> Dict[str, Any]:
    """List checkpoints for a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return {"object": "list", "data": [], "has_more": False}


@router.post("/fine_tuning/jobs/{job_id}/pause", response_model=FineTuningJobResponse)
async def pause_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
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

    return FineTuningJobResponse.from_entity(updated_job)


@router.post("/fine_tuning/jobs/{job_id}/resume", response_model=FineTuningJobResponse)
async def resume_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    """Resume a paused fine-tuning job."""

    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status == "queued":
        # Use the already normalized hyperparameters from the job
        hyperparameters = job.hyperparameters if job.hyperparameters else {}

        await enqueue(
            job_type="training",
            payload={
                "model": job.model,
                "training_file": job.training_file,
                "job_id": job.id,
                "hyperparameters": hyperparameters,
                "suffix": job.suffix or "custom",
                "config": {
                    "data_dir": settings.data_dir,
                    "checkpoint_dir": settings.fine_tuning_checkpoint_dir,
                    "checkpoint_interval": settings.fine_tuning_checkpoint_interval,
                    "max_checkpoints": settings.fine_tuning_max_checkpoints,
                    "webhook_url": settings.webhook_url,
                },
            },
            max_attempts=3,
        )
        await update_job_status(job_id, "queued")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot resume job {job_id}, current status: {job.status}")

    updated_job = await get_job(job_id)

    return FineTuningJobResponse.from_entity(updated_job)
