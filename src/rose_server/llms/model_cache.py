"""Simple model cache for reusing loaded models."""

import asyncio
from typing import Dict

from .llm import LLM

# Module-private cache
_models: Dict[str, LLM] = {}
_lock = asyncio.Lock()


async def get_model(model_id: str, config: Dict) -> LLM:
    """Get a cached model or create a new one."""
    async with _lock:
        if model_id not in _models:
            _models[model_id] = LLM(config)
        return _models[model_id]


def _cleanup_model(model: LLM) -> None:
    """Internal helper to clean up a model instance."""
    model.cleanup()


async def clear_model(model_id: str) -> None:
    """Remove a specific model from cache and clean up."""
    async with _lock:
        if model_id in _models:
            model = _models.pop(model_id)
            _cleanup_model(model)


async def clear_all() -> None:
    """Clear all cached models."""
    async with _lock:
        # Pop all models at once while holding the lock
        models_to_cleanup = list(_models.values())
        _models.clear()

    # Clean up models outside the lock
    for model in models_to_cleanup:
        _cleanup_model(model)
