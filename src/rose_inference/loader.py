"""Model loading utilities."""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch import quantization
from transformers import AutoTokenizer

from rose_inference.model_loader import load_hf_model, load_peft_model

logger = logging.getLogger(__name__)


async def load_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer for inference."""
    model_name = config.get("model_name")
    model_path = config.get("model_path")

    if not model_name:
        raise ValueError("model_name is required in config")

    logger.info(f"Loading model: {model_name} (path: {model_path})")

    # Load the model - check if it's a fine-tuned model by presence of model_path
    if model_path:
        # Fine-tuned models use PEFT loader
        model = load_peft_model(
            model_id=model_name,
            model_path=model_path,
            torch_dtype=config.get("torch_dtype"),
        )
    else:
        # Regular HF models
        model = load_hf_model(
            model_id=model_name,
            torch_dtype=config.get("torch_dtype"),
        )

    # Load tokenizer
    data_dir = config.get("data_dir", "./data")
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

    # Apply INT8 quantization if requested
    if config.get("quantization") == "int8":
        logger.info(f"Applying INT8 quantization to {model_name}")

        # Check if running on Apple Silicon
        if torch.backends.mps.is_available():
            # Use PyTorch's dynamic quantization for CPU/Apple Silicon
            # This converts weights to INT8 and uses INT8 compute where possible
            model = quantization.quantize_dynamic(  # type: ignore[attr-defined]
                model,
                {torch.nn.Linear},  # Quantize Linear layers
                dtype=torch.qint8,
            )
            logger.info("Applied INT8 dynamic quantization (Apple Silicon compatible)")
        elif torch.cuda.is_available():
            # For CUDA, could use bitsandbytes if available
            logger.warning(
                f"INT8 quantization requested for {model_name} but PyTorch dynamic quantization "
                "is not supported on CUDA. Skipping quantization. Consider using bitsandbytes "
                "for CUDA quantization support."
            )
        else:
            # CPU without MPS
            logger.warning(
                f"INT8 quantization requested for {model_name} but running on CPU without Apple Silicon MPS. "
                "Skipping quantization."
            )

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
