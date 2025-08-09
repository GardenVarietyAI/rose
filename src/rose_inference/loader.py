"""Model loading utilities."""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelLoaderParams:
    model_id: str
    model_name: str
    model_path: str
    torch_dtype: Optional[str]
    data_dir: str


def load_tokenizer(model_name: str, data_dir: str = "./data") -> Any:
    """Load tokenizer, trying local path first then downloading if needed."""
    models_dir = Path(data_dir) / "models"
    safe_model_name = model_name.replace("/", "--")
    local_model_path = models_dir / safe_model_name

    # Try local path first, then model name
    if local_model_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True)  # type: ignore[no-untyped-call]
    else:
        # Try loading from model name (will download if needed)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)  # type: ignore[no-untyped-call]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(params: ModelLoaderParams) -> Any:
    """Load a model and tokenizer for inference."""
    logger.info(f"Loading model: {params.model_name} (path: {params.model_path})")

    if params.model_path:
        model = load_peft_model(
            model_id=params.model_name,
            model_path=params.model_path,
            torch_dtype=params.torch_dtype,
        )
    else:
        model = load_hf_model(model_id=params.model_name, torch_dtype=params.torch_dtype)

    return model


def get_torch_dtype(dtype_str: Optional[str] = None) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_optimal_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hf_model(model_id: str, torch_dtype: Optional[str] = None) -> Any:
    """Load a HuggingFace model."""
    device = get_optimal_device()
    dtype = get_torch_dtype(torch_dtype)

    logger.info(f"Loading HF model {model_id} on {device} with dtype {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        local_files_only=False,
    )

    if device.type in {"cpu", "mps"}:
        model.to(device)  # type: ignore[arg-type]

    return model


def load_peft_model(model_id: str, model_path: str, torch_dtype: Optional[str] = None) -> Any:
    """Load a PEFT/LoRA fine-tuned model."""
    device = get_optimal_device()
    dtype = get_torch_dtype(torch_dtype)

    logger.info(f"Loading PEFT model {model_id} from {model_path} on {device} with dtype {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        local_files_only=True,
    )

    if device.type in {"cpu", "mps"}:
        model.to(device)  # type: ignore[arg-type]

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

    # GPU memory cleanup
    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                # Release cached memory on each CUDA device and wait for
                # pending kernels to finish so memory is actually freed.
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # type: ignore[no-untyped-call]
                torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        # Ensure all queued work is done before continuing to avoid
        # holding references to freed memory on Apple MPS.
        torch.mps.synchronize()

    gc.collect()
    logger.info("Memory cleanup completed")
