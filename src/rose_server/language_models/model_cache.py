"""Simple model cache for reusing loaded models."""

import asyncio
from typing import Dict

from rose_core.models import cleanup_model_memory

from .huggingface_llm import HuggingFaceLLM

# Module-private cache
_models: Dict[str, HuggingFaceLLM] = {}
_lock = asyncio.Lock()


async def get_model(model_id: str, config: Dict) -> HuggingFaceLLM:
    """Get a cached model or create a new one."""
    async with _lock:
        if model_id not in _models:
            _models[model_id] = HuggingFaceLLM(config)
        return _models[model_id]


async def clear_model(model_id: str) -> None:
    """Remove a specific model from cache and clean up."""
    async with _lock:
        if model_id in _models:
            model = _models.pop(model_id)
            if hasattr(model, "cleanup"):
                model.cleanup()
            else:
                model._model = None
                model._tokenizer = None
                cleanup_model_memory()


async def clear_all() -> None:
    """Clear all cached models."""
    async with _lock:
        model_ids = list(_models.keys())
        for model_id in model_ids:
            await clear_model(model_id)
