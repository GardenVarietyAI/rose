"""Model loading utilities."""

import logging
from typing import Any, Callable, Dict

import torch
from torch import quantization

from rose_core.models import get_tokenizer, load_hf_model

logger = logging.getLogger(__name__)


async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer for inference."""
    logger.info(f"Loading model: {model_name}")

    loader: Callable[..., Any] = model_config.get("loader", load_hf_model)

    # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
    model_id = model_config.get("model_path") or model_config.get("model_name", model_name)

    # Load the model based on loader type
    if loader.__name__ == "load_peft_model":
        # PEFT models require model_path
        model = loader(
            model_id=model_id,
            model_path=model_config.get("model_path"),
            torch_dtype=model_config.get("torch_dtype"),
        )
    else:
        # Regular HF models only need model_id
        model = loader(
            model_id=model_id,
            torch_dtype=model_config.get("torch_dtype"),
        )

    # Load tokenizer
    tokenizer = get_tokenizer(model_id)

    # Apply INT8 quantization if requested
    if model_config.get("quantization") == "int8":
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
        "config": model_config,
        "device": str(model.device) if hasattr(model, "device") else "cpu",
        "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
        "quantized": model_config.get("quantization") == "int8" and torch.backends.mps.is_available(),
    }
