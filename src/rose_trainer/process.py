"""Training job processor."""

import gc
import logging
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from pydantic import ValidationError

from rose_trainer.client import ServiceClient
from rose_trainer.fine_tuner import train
from rose_trainer.types import Hyperparameters

logger = logging.getLogger(__name__)


def unload_model(model: Optional[Any] = None) -> None:
    # Clean up specific model if provided
    if model is not None:
        try:
            # Check if it's a PEFT model
            if isinstance(model, PeftModel):
                logger.info("Cleaning up PEFT model")

                # Get the base model reference
                base_model = getattr(model, "base_model", None)

                # Clear adapter modules
                if hasattr(model, "peft_modules"):
                    for module_name in list(model.peft_modules.keys()):
                        try:
                            delattr(model, module_name)
                        except AttributeError:
                            pass

                # Delete the PEFT model
                del model

                # Clean up base model if it exists
                if base_model is not None:
                    del base_model

            else:
                # Regular model cleanup
                logger.info("Cleaning up regular model")
                del model

        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
        # Force synchronization
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()

    # MPS (Apple Silicon) cleanup
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()

    # Final garbage collection
    gc.collect()

    logger.info("Memory cleanup completed")


def process_training_job(job_id: str, payload: Dict[str, Any], client: ServiceClient) -> None:
    ft_job_id: str = payload["job_id"]
    model_name: str = payload["model"]
    training_file: str = payload["training_file"]
    hyperparameters: Dict[str, Any] = payload.get("hyperparameters", {})
    suffix: Optional[str] = payload["suffix"]
    config: Dict[str, Any] = payload.get("config", {})
    trainer: Optional[str] = payload.get("trainer")

    logger.info(f"Starting training job {job_id} for fine-tuning {ft_job_id} with trainer: {trainer or 'huggingface'}")

    # Create event callback for progress reporting
    def event_callback(level: str, msg: str, data: Dict[str, Any] | None = None) -> None:
        if level in ["info", "warning", "error"]:
            client.add_fine_tuning_event(ft_job_id, level, msg, data)

    def check_cancel_callback() -> str:
        status: str = client.check_fine_tuning_job_status(ft_job_id)
        return status

    try:
        # Update job status to running
        client.update_fine_tuning_job_status(ft_job_id, "running")

        # Add suffix to hyperparameters if provided
        if suffix:
            hyperparameters = hyperparameters.copy()
            hyperparameters["suffix"] = suffix

        try:
            hp = Hyperparameters(**hyperparameters)
        except ValidationError as ve:
            logger.error(f"Validation error for hyperparameters: {ve}")
            client.update_fine_tuning_job_status(
                ft_job_id, "failed", error={"message": f"Validation error: {ve}", "code": "validation_error"}
            )
            raise

        result = train(
            job_id=ft_job_id,
            model_name=model_name,
            training_file=training_file,
            hyperparameters=hp,
            client=client,
            check_cancel_callback=check_cancel_callback,
            event_callback=event_callback,
            config=config,
            trainer=trainer,
        )

        # Handle cancellation
        if result.get("cancelled"):
            client.update_fine_tuning_job_status(ft_job_id, "cancelled")
        else:
            # Build training metrics
            training_metrics = {
                "final_loss": result.get("final_loss"),
                "steps": result.get("steps"),
            }

            # Include perplexity if available (only when validation split was used)
            if result.get("final_perplexity") is not None:
                training_metrics["final_perplexity"] = result["final_perplexity"]

            # Include peak memory metrics if available
            if result.get("cuda_peak_memory_gb") is not None:
                training_metrics["cuda_peak_memory_gb"] = result["cuda_peak_memory_gb"]
            if result.get("mps_peak_memory_gb") is not None:
                training_metrics["mps_peak_memory_gb"] = result["mps_peak_memory_gb"]

            client.update_fine_tuning_job_status(
                ft_job_id,
                "succeeded",
                fine_tuned_model=result["model_name"],
                trained_tokens=int(result.get("tokens_processed", 0)),
                training_metrics=training_metrics,
            )

    except Exception as e:
        logger.exception(f"Training job {job_id} failed with unexpected exception")
        client.update_fine_tuning_job_status(ft_job_id, "failed", error={"message": str(e), "code": "job_error"})
        raise
    finally:
        unload_model()
