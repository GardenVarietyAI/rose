"""Common HuggingFace model loading utilities."""
import logging
import os
from typing import Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config import ServiceConfig
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_id: str,
    model_path: Optional[str] = None,
    device: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    offload_dir: Optional[str] = None,
    device_map: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load HuggingFace model and tokenizer.
    Args:
        model_id: Model identifier
        model_path: Optional local path for fine-tuned models
        device: Device to load on ("auto", "cuda", "cpu", "mps")
        torch_dtype: Torch dtype for model
        offload_dir: Directory for offloading model weights
        device_map: Optional device map for model parallelism
    Returns:
        Tuple of (model, tokenizer)
    """
    if model_path and os.path.exists(model_path):
        source_path = os.path.abspath(model_path)
        model_subdir = os.path.join(source_path, "model")
        if os.path.exists(model_subdir):
            source_path = model_subdir
        logger.info(f"Loading model from local path: {source_path}")
    else:
        source_path = model_id
        logger.info(f"Loading model from HuggingFace hub: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if offload_dir is None and device == "auto" and torch.cuda.is_available():
        offload_dir = os.path.join(ServiceConfig.MODEL_OFFLOAD_DIR, model_id.replace("/", "_"))
    if offload_dir:
        os.makedirs(offload_dir, exist_ok=True)
    if device_map is not None:
        final_device_map = device_map
    elif device == "auto":
        final_device_map = "auto"
    else:
        final_device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        source_path,
        torch_dtype=torch_dtype,
        device_map=final_device_map,
        trust_remote_code=True,
        offload_folder=offload_dir if offload_dir and final_device_map else None,
        offload_state_dict=bool(offload_dir and final_device_map),
        low_cpu_mem_usage=False if final_device_map is None else True,
    )
    if final_device_map is None and device not in ("auto", "cuda"):
        model = model.to(device)
    logger.info(f"Successfully loaded model: {model_id}")
    return model, tokenizer

def cleanup_model_memory():
    """Clean up model memory and cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
        gc.collect()