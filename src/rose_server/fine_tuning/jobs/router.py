import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from rose_server.config.settings import settings
from rose_server.entities.fine_tuning import FineTuningJob
from rose_server.fine_tuning.events.store import add_event
from rose_server.fine_tuning.jobs.store import create_job, get_job, list_jobs, list_jobs_by_status, update_job_status
from rose_server.models.store import create as create_language_model
from rose_server.schemas.fine_tuning import (
    FineTuningJobCreateRequest,
    FineTuningJobEventRequest,
    FineTuningJobResponse,
    FineTuningJobStatusUpdateRequest,
    Hyperparameters,
)

router = APIRouter(prefix="/jobs")
logger = logging.getLogger(__name__)


@router.post("", response_model=FineTuningJobResponse)
async def create_fine_tuning_job(request: FineTuningJobCreateRequest) -> FineTuningJobResponse:
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
        logger.info(f"Fine-tuning job {job.id} created and queued for direct worker processing")
        return FineTuningJobResponse.from_entity(job)
    except Exception as e:
        logger.error(f"Error creating fine-tuning job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=dict)
async def list_fine_tuning_jobs(
    limit: int = Query(default=20, ge=1, le=100, description="Number of jobs to retrieve"),
    after: Optional[str] = Query(default=None, description="Pagination cursor"),
) -> Dict[str, Any]:
    jobs = await list_jobs(limit=limit, after=after)
    return {
        "object": "list",
        "data": [FineTuningJobResponse.from_entity(job) for job in jobs],
        "has_more": len(jobs) == limit,
    }


@router.get("/queue")
async def get_queued_jobs(limit: int = Query(10, description="Max jobs to return")) -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": job.id,
                "type": "training",
                "status": job.status,
                "payload": {
                    "model": job.model,
                    "training_file": job.training_file,
                    "job_id": job.id,
                    "hyperparameters": job.hyperparameters or {},
                    "suffix": job.suffix or "custom",
                    "trainer": job.trainer,
                    "config": {
                        "data_dir": settings.data_dir,
                        "checkpoint_dir": settings.fine_tuning_checkpoint_dir,
                        "checkpoint_interval": settings.fine_tuning_checkpoint_interval,
                        "max_checkpoints": settings.fine_tuning_max_checkpoints,
                    },
                },
                "created_at": job.created_at,
            }
            for job in await list_jobs_by_status("queued", limit=limit)
        ],
    }


@router.get("/{job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return FineTuningJobResponse.from_entity(job)


@router.post("/{job_id}/cancel", response_model=FineTuningJobResponse)
async def cancel_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status in ["queued", "running"]:
        await update_job_status(job_id, "cancelled")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job {job_id} in status {job.status}")
    updated_job = await get_job(job_id)

    return FineTuningJobResponse.from_entity(updated_job)


@router.get("/{job_id}/checkpoints", response_model=dict)
async def list_fine_tuning_job_checkpoints(job_id: str) -> Dict[str, Any]:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return {"object": "list", "data": [], "has_more": False}


@router.post("/{job_id}/pause", response_model=FineTuningJobResponse)
async def pause_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status == "running":
        await update_job_status(job_id, "paused")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot pause job {job_id} in status {job.status}")

    updated_job = await get_job(job_id)

    return FineTuningJobResponse.from_entity(updated_job)


@router.post("/{job_id}/resume", response_model=FineTuningJobResponse)
async def resume_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status in ["paused", "failed"]:
        await update_job_status(job_id, "queued")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot resume job {job_id}, current status: {job.status}")

    updated_job = await get_job(job_id)

    return FineTuningJobResponse.from_entity(updated_job)


@router.patch("/{job_id}/status")
async def update_job_status_direct(job_id: str, request: FineTuningJobStatusUpdateRequest) -> FineTuningJobResponse:
    job = await update_job_status(
        job_id=job_id,
        status=request.status,
        error=request.error,
        fine_tuned_model=request.fine_tuned_model,
        trained_tokens=request.trained_tokens,
        training_metrics=request.training_metrics,
    )
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if request.status == "succeeded" and request.fine_tuned_model:
        try:
            model_path = Path("models") / request.fine_tuned_model

            created_model = await create_language_model(
                model_name=request.fine_tuned_model,
                path=str(model_path),
                parent=job.model,
                suffix=job.suffix,
            )

            logger.info(f"Registered fine-tuned model {request.fine_tuned_model} with ID {created_model.id}")

            updated_job = await update_job_status(
                job_id=job_id,
                status=request.status,
                fine_tuned_model=created_model.id,  # Use the generated ID instead of training name
                trained_tokens=request.trained_tokens,
                training_metrics=request.training_metrics,
            )
            if updated_job:
                job = updated_job

        except Exception as e:
            logger.error(f"Failed to register fine-tuned model {request.fine_tuned_model}: {e}")
            # Don't fail the status update if model registration fails

    return FineTuningJobResponse.from_entity(job)


@router.post("/{job_id}/events")
async def add_job_event(job_id: str, event: FineTuningJobEventRequest) -> Dict[str, Any]:
    await add_event(
        job_id=job_id,
        level=event.level,
        message=event.message,
        data=event.data,
    )
    return {"status": "accepted"}
