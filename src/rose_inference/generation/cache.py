"""Model caching and management for inference."""

import asyncio
import logging
from typing import Any, Dict

from rose_core.models import cleanup_model_memory, get_tokenizer, load_hf_model

logger = logging.getLogger(__name__)

# Global model cache
_models: Dict[str, Dict[str, Any]] = {}
_cache_lock = asyncio.Lock()


async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer (cached)."""
    # Fast path - check without lock first
    if model_name in _models:
        logger.info(f"Using cached model: {model_name}")
        return _models[model_name]

    # Slow path - need to load model
    async with _cache_lock:
        # Double-check after acquiring lock (another task might have loaded it)
        if model_name in _models:
            logger.info(f"Using cached model: {model_name} (loaded by another task)")
            return _models[model_name]

        logger.info(f"Loading model: {model_name}")
        try:
            # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
            model_id = model_config.get("model_path") or model_config.get("model_name", model_name)
            model = load_hf_model(
                model_id=model_id,
                torch_dtype=model_config.get("torch_dtype"),
            )

            # Load tokenizer
            tokenizer = get_tokenizer(model_config.get("model_path") or model_config.get("model_name", model_name))

            _models[model_name] = {"model": model, "tokenizer": tokenizer, "config": model_config}

            logger.info(f"Successfully loaded model: {model_name}")
            return _models[model_name]

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise


def cleanup_models():
    """Clean up cached models on exit."""
    logger.info("Shutting down inference server, cleaning up models...")
    _models.clear()
    cleanup_model_memory()
    logger.info("Cleanup complete")
