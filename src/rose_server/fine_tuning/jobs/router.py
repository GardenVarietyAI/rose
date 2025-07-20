import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from rose_server.config.settings import settings
from rose_server.entities.fine_tuning import FineTuningJob
from rose_server.fine_tuning.jobs.store import create_job, get_job, list_jobs, update_job_status
from rose_server.queues.store import enqueue, find_job_by_payload_field, request_cancel, request_pause
from rose_server.schemas.fine_tuning import (
    FineTuningJobCreateRequest,
    FineTuningJobResponse,
    Hyperparameters,
)

router = APIRouter(prefix="/jobs")
logger = logging.getLogger(__name__)


@router.post("", response_model=FineTuningJobResponse)
async def create_fine_tuning_job(request: FineTuningJobCreateRequest) -> FineTuningJobResponse:
    """Create a fine-tuning job."""
    try:
        # Extract hyperparameters from method or request
        method_config = getattr(request.method, request.method.type, None)
        if method_config and method_config.hyperparameters:
            hp_dict = method_config.hyperparameters.model_dump()
        elif request.hyperparameters:
            hp_dict = (
                request.hyperparameters
                if isinstance(request.hyperparameters, dict)
                else request.hyperparameters.model_dump()
            )
        else:
            hp_dict = {}

        # Convert "auto" values to actual values
        if hp_dict.get("batch_size") == "auto":
            hp_dict["batch_size"] = settings.fine_tuning_auto_batch_size
        if hp_dict.get("learning_rate_multiplier") == "auto":
            hp_dict["learning_rate_multiplier"] = settings.fine_tuning_auto_learning_rate_multiplier
        if hp_dict.get("n_epochs") == "auto":
            hp_dict["n_epochs"] = settings.fine_tuning_auto_epochs

        # Create Hyperparameters object
        try:
            hyperparameters_obj = Hyperparameters(**hp_dict)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Convert to dict
        hyperparameters = hyperparameters_obj.model_dump(exclude_none=True)

        # Calculate learning rate if using multiplier
        if hyperparameters.get("learning_rate_multiplier") is not None and "learning_rate" not in hyperparameters:
            hyperparameters["base_learning_rate"] = settings.fine_tuning_base_learning_rate
            hyperparameters["learning_rate"] = (
                hyperparameters["base_learning_rate"] * hyperparameters["learning_rate_multiplier"]
            )

        # Apply training defaults
        training_defaults = {
            "max_length": 512,
            "gradient_accumulation_steps": 1,
            "validation_split": 0.1,
            "early_stopping_patience": 3,
            "warmup_ratio": 0.1,
            "scheduler_type": "cosine",
            "min_lr_ratio": 0.1,
            "weight_decay": 0.01,
            "use_lora": True,
            "seed": request.seed or 42,
            "suffix": request.suffix or "custom",
            # Use from settings if available
            "eval_batch_size": settings.fine_tuning_eval_batch_size,
        }

        for key, value in training_defaults.items():
            if key not in hyperparameters:
                hyperparameters[key] = value

        # Preserve eval_metrics from hyperparameters_obj
        hyperparameters["eval_metrics"] = hyperparameters_obj.eval_metrics

        # TODO: We skip the "validating_files" status for now since we don't actually validate the JSONL format.
        # In the future, we should validate that the file exists and contains properly formatted training data.
        job = await create_job(
            FineTuningJob(
                model=request.model,
                status="queued",
                training_file=request.training_file,
                validation_file=request.validation_file,
                seed=request.seed or 42,
                suffix=request.suffix,
                meta=request.metadata,
                hyperparameters=hyperparameters,
                method=request.method.model_dump() if request.method else None,
                trainer=request.trainer or "huggingface",
            )
        )

        await enqueue(
            job_type="training",
            payload={
                "model": request.model,
                "training_file": request.training_file,
                "job_id": job.id,
                "hyperparameters": hyperparameters,
                "suffix": request.suffix or "custom",
                "trainer": job.trainer,  # Pass the trainer from the created job
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


@router.get("", response_model=dict)
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


@router.get("/{job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    """Retrieve a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return FineTuningJobResponse.from_entity(job)


@router.post("/{job_id}/cancel", response_model=FineTuningJobResponse)
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


@router.get("/{job_id}/checkpoints", response_model=dict)
async def list_fine_tuning_job_checkpoints(job_id: str) -> Dict[str, Any]:
    """List checkpoints for a fine-tuning job."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return {"object": "list", "data": [], "has_more": False}


@router.post("/{job_id}/pause", response_model=FineTuningJobResponse)
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


@router.post("/{job_id}/resume", response_model=FineTuningJobResponse)
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
