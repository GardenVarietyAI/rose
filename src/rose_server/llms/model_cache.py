"""Simple model cache for reusing loaded models."""

from typing import Any, Dict

from .llm import LLM

# Module-private cache for config objects
# NOTE: Currently just caches lightweight config wrappers, but kept for:
# 1. Future tokenizer caching (loading tokenizers is expensive)
# 2. Consistency with inference server architecture
# 3. Potential caching of token counts or other computed properties
_models: Dict[str, LLM] = {}


async def get_model(model_id: str, config: Dict[str, Any]) -> LLM:
    """Get a cached model or create a new one."""
    if model_id not in _models:
        _models[model_id] = LLM(config)
    return _models[model_id]


async def clear_model(model_id: str) -> None:
    """Remove a specific model from cache."""
    if model_id in _models:
        _models.pop(model_id)


async def clear_all() -> None:
    """Clear all cached models."""
    _models.clear()
