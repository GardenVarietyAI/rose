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


async def clear_model(model_id: str) -> None:
    """Remove a specific model from cache."""
    async with _lock:
        if model_id in _models:
            _models.pop(model_id)


async def clear_all() -> None:
    """Clear all cached models."""
    async with _lock:
        _models.clear()
