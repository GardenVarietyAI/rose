"""Model caching and management for inference."""

import asyncio
import gc
import logging
from typing import Any, Dict

from rose_core.models import cleanup_model_memory, get_tokenizer, loader

logger = logging.getLogger(__name__)


class ModelCache:
    """Simple cache that keeps only the last used model."""

    def __init__(self) -> None:
        self._current_model: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def load_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load a model and tokenizer (cached last model only)."""
        # Fast path - check if it's the current model
        if self._current_model.get("name") == model_name:
            logger.info(f"Using cached model: {model_name}")
            return self._current_model

        # Need to load new model
        old = None
        async with self._lock:
            # Double-check after acquiring lock
            if self._current_model.get("name") == model_name:
                logger.info(f"Using cached model: {model_name} (loaded by another task)")
                return self._current_model

            old = self._current_model  # Keep pointer to the old dict
            logger.info(f"Loading model: {model_name}")

            try:
                # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
                model_id = model_config.get("model_path") or model_config.get("model_name", model_name)
                model = loader(
                    model_id=model_id,
                    model_path=model_config.get("model_path"),
                    torch_dtype=model_config.get("torch_dtype"),
                )

                # Load tokenizer
                tokenizer = get_tokenizer(model_id)

                # Replace dict instead of mutating/clearing
                self._current_model = {
                    "name": model_name,
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": model_config,
                    "device": str(model.device) if hasattr(model, "device") else "cpu",
                    "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
                }

            except Exception as e:
                # Restore old model on failure
                if old:
                    self._current_model = old
                    logger.warning(f"Restored previous model after load failure: {old.get('name')}")
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

        # Clean up *after* lock is released so loading thread isn't blocked
        if old:
            old_name = old.get("name", "unknown")
            logger.info(f"Cleaning up memory from previous model: {old_name}")
            del old  # Drop our reference
            gc.collect()  # Free CPU-side objects predictably
            cleanup_model_memory()

        return self._current_model

    def cleanup(self) -> None:
        """Clean up cached models."""
        logger.info("Cleaning up cached models...")
        self._current_model = {}
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
