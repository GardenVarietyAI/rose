"""Enhanced model cleanup for PEFT/LoRA models."""

import gc
import logging
from typing import Any, Optional

import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


def cleanup_peft_model(model: Any) -> None:
    """Properly cleanup PEFT/LoRA models before general cleanup.

    Args:
        model: The model to cleanup (can be PeftModel or regular model)
    """
    if model is None:
        return

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

        # Force garbage collection
        gc.collect()

    except Exception as e:
        logger.warning(f"Error during model cleanup: {e}")


def cleanup_model_memory(model: Optional[Any] = None) -> None:
    """Enhanced cleanup that handles PEFT models properly.

    Args:
        model: Optional model to cleanup before general memory cleanup
    """
    # Clean up specific model if provided
    if model is not None:
        cleanup_peft_model(model)

    # Run general garbage collection
    gc.collect()

    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # type: ignore[attr-defined]
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
