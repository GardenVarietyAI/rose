"""Model loading utilities."""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from torch import quantization
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str, data_dir: str = "./data") -> Any:
    """Load tokenizer, trying local path first then downloading if needed."""
    models_dir = Path(data_dir) / "models"
    safe_model_name = model_name.replace("/", "--")
    local_model_path = models_dir / safe_model_name

    # Try local path first, then model name
    if local_model_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True)
    else:
        # Try loading from model name (will download if needed)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


async def load_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer for inference."""
    model_name = config.get("model_name")
    model_path = config.get("model_path")

    if not model_name:
        raise ValueError("model_name is required in config")

    logger.info(f"Loading model: {model_name} (path: {model_path})")

    if model_path:
        model = load_peft_model(model_id=model_name, model_path=model_path, torch_dtype=config.get("torch_dtype"))
    else:
        model = load_hf_model(model_id=model_name, torch_dtype=config.get("torch_dtype"))

    # Load tokenizer
    tokenizer = load_tokenizer(model_name, config.get("data_dir", "./data"))

    # Apply INT8 quantization if requested
    if config.get("quantization") == "int8":
        if torch.backends.mps.is_available():
            logger.warning("Skipping INT8 dynamic quantization on Apple Silicon")
        elif torch.cuda.is_available():
            logger.warning("Skipping INT8 dynamic quantization on CUDA, install bitsandbytes")
        else:
            # Use PyTorch's dynamic quantization for CPU
            # This converts weights to INT8 and uses INT8 compute where possible
            model = quantization.quantize_dynamic(  # type: ignore[attr-defined]
                model,
                {torch.nn.Linear},  # Quantize Linear layers
                dtype=torch.qint8,
            )
            logger.info(f"INT8 quantization requested for {model_name} on CPU")

    # Return model info
    return {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "device": str(model.device) if hasattr(model, "device") else "cpu",
        "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
        "quantized": config.get("quantization") == "int8" and not torch.backends.mps.is_available(),
    }


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


def unload_model(model: Optional[Any] = None) -> None:
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
