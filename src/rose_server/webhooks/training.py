"""Training job webhook handlers."""

import logging
from pathlib import Path

from rose_server.fine_tuning.events.store import add_event
from rose_server.fine_tuning.jobs.store import get_job, mark_job_failed, update_job_result_files, update_job_status
from rose_server.models.store import create as create_language_model
from rose_server.queues.store import update_job_status as update_queue_job_status
from rose_server.schemas.webhooks import WebhookEvent
from rose_server.webhooks.results_output import create_result_file

logger = logging.getLogger(__name__)


async def handle_training_webhook(event: WebhookEvent, registry) -> dict:
    """Handle training job webhooks."""
    if event.event == "job.completed":
        await _handle_training_completed(event, registry)
    elif event.event == "job.failed":
        await _handle_training_failed(event)
    elif event.event == "job.progress":
        await _handle_training_progress(event)
    elif event.event == "job.running":
        await _handle_training_running(event)
    elif event.event == "job.cancelled":
        await _handle_training_cancelled(event)
    else:
        logger.warning(f"Unknown training webhook event: {event.event}")
    return {"status": "ok", "message": "Training webhook processed"}


async def _handle_training_completed(event: WebhookEvent, registry) -> None:
    """Handle successful training job completion."""
    result_file_id = await _create_training_result_file(event)

    # Get the fine-tuning job to access base model info
    ft_job = await get_job(event.object_id)
    if not ft_job:
        logger.error(f"Fine-tuning job {event.object_id} not found")
        return

    fine_tuned_model = event.data.get("fine_tuned_model")

    if result_file_id:
        await update_job_result_files(event.object_id, [result_file_id])

    # Register the fine-tuned model in the LanguageModel table FIRST
    if not fine_tuned_model:
        logger.error(f"Fine-tuning model from job {event.object_id} not found")
        return

    try:
        model_path = Path("models") / fine_tuned_model
        base_config = await registry.get_model_config(ft_job.model)
        model_name = base_config.model_name if base_config else None

        created_model = await create_language_model(
            model_name=model_name or ft_job.model,
            name=fine_tuned_model,
            path=str(model_path),
            parent=ft_job.model,
            suffix=getattr(ft_job, "suffix", None),
        )

        logger.info(f"Registered fine-tuned model {fine_tuned_model} with ID {created_model.id} in database")

        # Extract training metrics - only include non-None values
        training_metrics = {}
        metric_keys = [
            "final_loss",
            "steps",
            "trained_tokens",
            "final_perplexity",
            "cuda_peak_memory_gb",
            "mps_peak_memory_gb",
        ]
        for key in metric_keys:
            value = event.data.get(key)
            if value is not None:
                training_metrics[key] = value

        # Update job with the UUID instead of the timestamp-based name
        await update_job_status(
            event.object_id,
            status="succeeded",
            fine_tuned_model=created_model.id,  # Use UUID here
            trained_tokens=event.data.get("trained_tokens", 0),
            training_metrics=training_metrics if training_metrics else None,
        )

        await _update_queue_job_completed(event)
    except Exception as e:
        logger.error(f"Failed to register fine-tuned model: {e}")


async def _handle_training_failed(event: WebhookEvent) -> None:
    """Handle failed training job."""
    error_msg = event.data.get("error", {}).get("message", "Unknown error")
    await mark_job_failed(event.object_id, error_msg)
    await _update_queue_job_failed(event, error_msg)


async def _create_training_result_file(event: WebhookEvent) -> str | None:
    """Create result file for completed training job."""
    final_loss = event.data.get("final_loss")
    steps = event.data.get("steps")
    final_perplexity = event.data.get("final_perplexity")  # May be None if no validation split
    if final_loss is None or steps is None:
        logger.warning(f"Missing final_loss or steps for job {event.object_id}")
        return None
    try:
        result_file_id = await create_result_file(event.object_id, final_loss, steps, final_perplexity)
        logger.info(f"Created result file {result_file_id} for job {event.object_id}")
        return result_file_id
    except Exception as e:
        logger.error(f"Failed to create result file: {e}")
        return None


async def _update_queue_job_completed(event: WebhookEvent) -> None:
    """Update queue job status to completed."""
    await update_queue_job_status(event.job_id, "completed", result=event.data)


async def _update_queue_job_failed(event: WebhookEvent, error_msg: str) -> None:
    """Update queue job status to failed."""
    # Note: The fail_job method in queues store handles retry logic,
    # but for webhooks we want to directly mark as failed
    await update_queue_job_status(event.job_id, "failed", result={"error": error_msg})


async def _handle_training_progress(event: WebhookEvent) -> None:
    """Handle training progress events."""
    message = event.data.get("message", "Training progress")
    level = event.data.get("level", "info")
    await add_event(job_id=event.object_id, level=level, message=message, data=event.data)
    logger.debug(f"Added progress event for job {event.object_id}: {message}")


async def _handle_training_running(event: WebhookEvent) -> None:
    """Handle training job starting to run."""
    await update_job_status(event.object_id, status="running")
    await update_queue_job_status(event.job_id, "running")
    logger.info(f"Training job {event.object_id} is now running")


async def _handle_training_cancelled(event: WebhookEvent) -> None:
    """Handle training job cancellation."""
    await update_job_status(event.object_id, status="cancelled")
    await update_queue_job_status(event.job_id, "cancelled")
    logger.info(f"Training job {event.object_id} was cancelled")
