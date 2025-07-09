"""Model loading utilities for training."""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def get_local_model_path(model_id: str, data_dir: str = "./data") -> Optional[Path]:
    """Get the local path for a downloaded model if it exists.

    Args:
        model_id: HuggingFace model identifier
        data_dir: Data directory path

    Returns:
        Path to local model directory if it exists, None otherwise
    """
    models_dir = Path(data_dir) / "models"
    safe_model_name = model_id.replace("/", "--")
    local_model_path = models_dir / safe_model_name

    if local_model_path.exists():
        return local_model_path
    return None


def get_tokenizer(path: str, data_dir: str = "./data") -> PreTrainedTokenizerBase:
    """Setup tokenizer with proper padding token."""
    # Check if model is downloaded locally
    local_model_path = get_local_model_path(path, data_dir)

    if local_model_path:
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer  # type: ignore[no-any-return]


def get_torch_dtype(torch_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """Get torch dtype, defaulting to float16 for CUDA and float32 for CPU."""
    if torch_dtype is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return torch_dtype


def get_optimal_device() -> str:
    """Get the optimal device for model inference/training."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_hf_model(
    model_id: str,
    config: Dict[str, Any],
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    """Load HuggingFace model.
    Args:
        model_id: Model identifier from HuggingFace hub
        config: Configuration dictionary containing data_dir
        torch_dtype: Torch dtype for model
    Returns:
        PreTrainedModel
    """
    data_dir = config.get("data_dir", "./data")

    # Check if model is downloaded locally
    local_model_path = get_local_model_path(model_id, data_dir)

    if local_model_path:
        logger.info(f"Loading model from local path: {local_model_path}")
        model_path = str(local_model_path)
        local_files_only = True
    else:
        logger.info(f"Loading model from HuggingFace hub: {model_id}")
        model_path = model_id
        local_files_only = True

    # Create offload directory
    offload_dir = os.path.join(data_dir, "offload", model_id.replace("/", "--"))
    os.makedirs(offload_dir, exist_ok=True)

    device = get_optimal_device()
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        torch_dtype=get_torch_dtype(torch_dtype),
        offload_folder=offload_dir,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = model.to(device)

    logger.info(f"Successfully loaded model: {model_id}")
    return model  # type: ignore[no-any-return]


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


def unload_model(model: Optional[Any] = None) -> None:
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
