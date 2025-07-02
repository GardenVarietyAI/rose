"""Model caching and management for inference."""

import asyncio
import logging
from typing import Any, Dict

from rose_core.models import cleanup_model_memory, get_tokenizer, load_hf_model

logger = logging.getLogger(__name__)


class ModelCache:
    """Simple cache that keeps only the last used model."""

    def __init__(self):
        self._current_model: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def load_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model and tokenizer (cached last model only)."""
        # Fast path - check if it's the current model
        if self._current_model.get("name") == model_name:
            logger.info(f"Using cached model: {model_name}")
            return self._current_model

        # Need to load new model
        old_model_name = None
        async with self._lock:
            # Double-check after acquiring lock
            if self._current_model.get("name") == model_name:
                logger.info(f"Using cached model: {model_name} (loaded by another task)")
                return self._current_model

            # Clear previous model if exists
            if self._current_model:
                old_model_name = self._current_model.get("name")
                logger.info(f"Clearing previous model: {old_model_name}")
                self._current_model.clear()

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

                self._current_model.update(
                    {"name": model_name, "model": model, "tokenizer": tokenizer, "config": model_config}
                )

                logger.info(f"Successfully loaded model: {model_name}")
                return self._current_model

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

        # Cleanup old model memory outside the lock
        if old_model_name:
            logger.info(f"Cleaning up memory from previous model: {old_model_name}")
            cleanup_model_memory()

    def cleanup(self) -> None:
        """Clean up cached models."""
        logger.info("Cleaning up cached models...")
        self._current_model.clear()
        cleanup_model_memory()
        logger.info("Cleanup complete")


# Global instance
_cache = ModelCache()


# Compatibility functions
async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer (compatibility wrapper)."""
    return await _cache.load_model(model_name, model_config)


def cleanup_models() -> None:
    """Clean up cached models on exit (compatibility wrapper)."""
    _cache.cleanup()
