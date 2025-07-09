"""Local model loading implementation without external dependencies."""

import gc
import logging
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def get_torch_dtype(dtype_str: Optional[str] = None) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_optimal_device() -> str:
    """Get the optimal device for model loading."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf_model(model_id: str, torch_dtype: Optional[str] = None) -> Any:
    """Load a HuggingFace model."""
    device = get_optimal_device()
    dtype = get_torch_dtype(torch_dtype)

    logger.info(f"Loading HF model {model_id} on {device} with dtype {dtype}")

    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=False,  # Allow downloading if needed
    )

    if device == "mps":
        model = model.to(device)

    return model


def load_peft_model(model_id: str, model_path: str, torch_dtype: Optional[str] = None) -> Any:
    """Load a PEFT/LoRA fine-tuned model."""
    device = get_optimal_device()
    dtype = get_torch_dtype(torch_dtype)

    logger.info(f"Loading PEFT model from {model_path}")

    # Load the PEFT model with its base model
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )

    if device == "mps":
        model = model.to(device)

    return model


def cleanup_model_memory(model: Optional[Any] = None) -> None:
    """Clean up model memory."""
    if model is not None:
        # Handle PEFT models specially
        if isinstance(model, PeftModel):
            logger.info("Cleaning up PEFT model")
            # Get base model reference
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

    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()
    logger.info("Memory cleanup completed")
