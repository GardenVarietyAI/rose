"""Common HuggingFace model loading utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rose_core.config.service import MODEL_OFFLOAD_DIR

logger = logging.getLogger(__name__)


def get_tokenizer(path: str) -> PreTrainedTokenizerBase:
    """Setup tokenizer with proper padding token."""
    tokenizer = AutoTokenizer.from_pretrained(path)
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


def _resolve_model_path(model_path: str) -> str:
    """Resolve model path, checking for 'model' subdirectory."""
    source_path = os.path.abspath(model_path)
    model_subdir = os.path.join(source_path, "model")
    if os.path.exists(model_subdir):
        source_path = model_subdir
    return source_path


def load_hf_model(
    model_id: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    """Load HuggingFace model.
    Args:
        model_id: Model identifier from HuggingFace hub
        torch_dtype: Torch dtype for model
    Returns:
        PreTrainedModel
    """
    logger.info(f"Loading model from HuggingFace hub: {model_id}")

    # Create offload directory
    offload_dir = os.path.join(MODEL_OFFLOAD_DIR, model_id.replace("/", "_"))
    os.makedirs(offload_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        torch_dtype=get_torch_dtype(torch_dtype),
        offload_folder=offload_dir,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device=get_optimal_device(),
    )

    logger.info(f"Successfully loaded model: {model_id}")
    return model  # type: ignore[no-any-return]


def load_peft_model(
    model_id: str,
    model_path: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> PeftModel:
    """Load PEFT/LoRA model from local path.
    Args:
        model_id: Model identifier (used for offload dir naming)
        model_path: Local path to the PEFT model (required)
        torch_dtype: Torch dtype for model
    Returns:
        PeftModel
    """
    if not model_path:
        raise ValueError("model_path is required for loading PEFT models")

    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")

    # Resolve model path
    source_path = _resolve_model_path(model_path)
    logger.info(f"Loading model from local path: {source_path}")

    # Check if this is a LoRA model by looking for adapter_config.json
    adapter_config_path = Path(source_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(f"No adapter_config.json found at {source_path}. This doesn't appear to be a PEFT model.")

    logger.info(f"Detected LoRA adapter at {source_path}")

    # Load adapter config to find base model
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    # Get base model name from adapter config
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"LoRA adapter at {source_path} missing base_model_name_or_path in adapter_config.json. "
            "Cannot load adapter without knowing which base model to use."
        )

    # Create offload directory
    offload_dir = os.path.join(MODEL_OFFLOAD_DIR, model_id.replace("/", "_"))
    os.makedirs(offload_dir, exist_ok=True)

    # Load base model first
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(  # type: ignore[no-untyped-call]
        base_model_name,
        torch_dtype=get_torch_dtype(torch_dtype),
        trust_remote_code=True,
        offload_folder=offload_dir,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        device=get_optimal_device(),
    )

    # Load and apply LoRA adapter
    logger.info(f"Loading LoRA adapter from {source_path}")
    model = PeftModel.from_pretrained(base_model, source_path)

    # Set to evaluation mode
    model.eval()

    logger.info(f"Successfully loaded PEFT model: {model_id}")
    return model


def cleanup_model_memory() -> None:
    """Clean up model memory and cache."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
        gc.collect()
