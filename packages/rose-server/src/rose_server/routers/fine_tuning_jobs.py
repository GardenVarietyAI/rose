import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from rose_server.database import current_timestamp, get_session
from rose_server.entities.fine_tuning import FineTuningEvent, FineTuningJob
from rose_server.entities.models import LanguageModel
from rose_server.schemas.fine_tuning import (
    FineTuningJobCreateRequest,
    FineTuningJobResponse,
    FineTuningJobStatusUpdateRequest,
    Hyperparameters,
)
from rose_server.services.fine_tuning_step_metrics import build_training_results
from rose_server.settings import settings
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

router = APIRouter(prefix="/v1/fine_tuning/jobs")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_STEPS_FALLBACK = 1000
MAX_EVENTS_LIMIT = 1000


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
        job = FineTuningJob(
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

        async with get_session() as session:
            session.add(job)
            await session.commit()
            await session.refresh(job)
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
    statement = select(FineTuningJob).order_by(FineTuningJob.created_at.desc())
    if after:
        statement = statement.where(FineTuningJob.id > after)
    statement = statement.limit(limit)
    async with get_session(read_only=True) as session:
        jobs = list((await session.execute(statement)).scalars().all())
    return {
        "object": "list",
        "data": [FineTuningJobResponse.from_entity(job) for job in jobs],
        "has_more": len(jobs) == limit,
    }


@router.get("/queue")
async def get_queued_jobs(limit: int = Query(10, description="Max jobs to return")) -> Dict[str, Any]:
    async with get_session(read_only=True) as session:
        jobs = list(
            (
                await session.execute(
                    select(FineTuningJob)
                    .where(FineTuningJob.status == "queued")
                    .order_by(FineTuningJob.created_at.asc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )

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
            for job in jobs
        ],
    }


@router.get("/{job_id}", response_model=FineTuningJobResponse)
async def retrieve_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return FineTuningJobResponse.from_entity(job)


@router.post("/{job_id}/cancel", response_model=FineTuningJobResponse)
async def cancel_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status in ["queued", "running"]:
        async with get_session() as session:
            job = await session.get(FineTuningJob, job_id)
            if job:
                job.status = "cancelled"
                job.finished_at = current_timestamp()
                session.add(job)
                await session.commit()
                await session.refresh(job)
                logger.info(f"Updated job {job_id} status to cancelled")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job {job_id} in status {job.status}")
    async with get_session(read_only=True) as session:
        updated_job = await session.get(FineTuningJob, job_id)

    return FineTuningJobResponse.from_entity(updated_job)


@router.get("/{job_id}/checkpoints", response_model=dict)
async def list_fine_tuning_job_checkpoints(job_id: str) -> Dict[str, Any]:
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")
    return {"object": "list", "data": [], "has_more": False}


@router.post("/{job_id}/pause", response_model=FineTuningJobResponse)
async def pause_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status == "running":
        async with get_session() as session:
            job = await session.get(FineTuningJob, job_id)
            if job:
                job.status = "paused"
                session.add(job)
                await session.commit()
                await session.refresh(job)
                logger.info(f"Updated job {job_id} status to paused")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot pause job {job_id} in status {job.status}")

    async with get_session(read_only=True) as session:
        updated_job = await session.get(FineTuningJob, job_id)

    return FineTuningJobResponse.from_entity(updated_job)


@router.post("/{job_id}/resume", response_model=FineTuningJobResponse)
async def resume_fine_tuning_job(job_id: str) -> FineTuningJobResponse:
    async with get_session(read_only=True) as session:
        job = await session.get(FineTuningJob, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found")

    if job.status in ["paused", "failed"]:
        async with get_session() as session:
            job = await session.get(FineTuningJob, job_id)
            if job:
                job.status = "queued"
                session.add(job)
                await session.commit()
                await session.refresh(job)
                logger.info(f"Updated job {job_id} status to queued")
    else:
        raise HTTPException(status_code=400, detail=f"Cannot resume job {job_id}, current status: {job.status}")

    async with get_session(read_only=True) as session:
        updated_job = await session.get(FineTuningJob, job_id)

    return FineTuningJobResponse.from_entity(updated_job)


@router.patch("/{job_id}/status")
async def update_job_status_direct(job_id: str, request: FineTuningJobStatusUpdateRequest) -> FineTuningJobResponse:
    async with get_session() as session:
        job = await session.get(FineTuningJob, job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if request.fine_tuned_model:
            job.fine_tuned_model = request.fine_tuned_model
        if request.trained_tokens:
            job.trained_tokens = request.trained_tokens
        if request.error:
            job.error = request.error
        if request.training_metrics:
            job.training_metrics = request.training_metrics

        job.status = request.status
        if request.status in ["succeeded", "failed", "cancelled"]:
            job.finished_at = current_timestamp()
        elif request.status == "running" and not job.started_at:
            job.started_at = current_timestamp()

        session.add(job)
        await session.commit()
        await session.refresh(job)
        logger.info(f"Updated job {job_id} status to {request.status}")

    if request.status == "succeeded" and request.fine_tuned_model:
        try:
            model_path = Path("models") / request.fine_tuned_model

            # Create the fine-tuned model
            unique_hash = uuid.uuid4().hex[:6]
            suffix_part = job.suffix if job.suffix else "default"
            flat_base_model = job.model.replace("/", "--")
            model_id = f"ft:{flat_base_model}:user:{suffix_part}:{unique_hash}"

            model = LanguageModel(
                id=model_id,
                model_name=request.fine_tuned_model,
                path=str(model_path),
                kind=None,
                is_fine_tuned=True,
                temperature=0.7,
                top_p=0.9,
                timeout=None,
                owned_by="user",
                parent=job.model,
                quantization=None,
                lora_target_modules=[],
            )

            async with get_session() as session:
                try:
                    session.add(model)
                    await session.commit()
                    await session.refresh(model)
                except IntegrityError:
                    # Model already exists, retrieve and return it
                    await session.rollback()
                    result = await session.execute(select(LanguageModel).where(LanguageModel.id == model_id))
                    model = result.scalar_one()

            logger.info(f"Registered fine-tuned model {request.fine_tuned_model} with ID {model.id}")

            async with get_session() as update_session:
                updated_job = await update_session.get(FineTuningJob, job_id)
                if updated_job:
                    updated_job.fine_tuned_model = model.id  # Use the generated ID instead of training name
                    if request.trained_tokens:
                        updated_job.trained_tokens = request.trained_tokens
                    if request.training_metrics:
                        updated_job.training_metrics = request.training_metrics
                    updated_job.status = request.status
                    if request.status in ["succeeded", "failed", "cancelled"]:
                        updated_job.finished_at = current_timestamp()
                    elif request.status == "running" and not updated_job.started_at:
                        updated_job.started_at = current_timestamp()
                    update_session.add(updated_job)
                    await update_session.commit()
                    await update_session.refresh(updated_job)
                    logger.info(f"Updated job {job_id} status to {request.status}")
            if updated_job:
                job = updated_job

        except Exception as e:
            logger.error(f"Failed to register fine-tuned model {request.fine_tuned_model}: {e}")
            # Don't fail the status update if model registration fails

    # Update detailed training results for successful jobs
    if request.status == "succeeded" and request.training_metrics:
        try:
            final_loss = request.training_metrics.get("final_loss", 0.0)
            steps = request.trained_tokens or DEFAULT_STEPS_FALLBACK
            final_perplexity = request.training_metrics.get("final_perplexity")

            # Get events for training metrics
            statement = (
                select(FineTuningEvent)
                .where(FineTuningEvent.job_id == job_id)
                .order_by(FineTuningEvent.created_at.asc())
            )
            async with get_session(read_only=True) as session:
                events = list((await session.execute(statement)).scalars().all())
            detailed_metrics = build_training_results(job, events, final_loss, steps, final_perplexity)
            if detailed_metrics:
                async with get_session() as update_session:
                    job_update = await update_session.get(FineTuningJob, job_id)
                    if job_update:
                        job_update.training_metrics = detailed_metrics
                        job_update.status = request.status
                        update_session.add(job_update)
                        await update_session.commit()
                        await update_session.refresh(job_update)
                        logger.info(f"Updated job {job_id} training metrics")
                logger.info(f"Training results updated for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to build training results for job {job_id}: {e}")
            # Don't fail the status update if training results build fails

    return FineTuningJobResponse.from_entity(job)
