"""Model loading utilities."""

import logging
from typing import Any, Callable, Dict

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

    # Return model info
    return {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "config": model_config,
        "device": str(model.device) if hasattr(model, "device") else "cpu",
        "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
    }
