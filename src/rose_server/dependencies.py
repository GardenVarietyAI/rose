from typing import Optional

from fastapi import Request

from rose_server._inference import InferenceServer
from rose_server.models.store import get as get_language_model
from rose_server.schemas.models import ModelConfig

__all__ = ["InferenceServer", "get_inference_server", "get_model_config"]


def get_inference_server(request: Request) -> InferenceServer:
    return request.app.state.inference_server


async def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """Get configuration for a model from the database."""
    model = await get_language_model(model_id)
    if not model:
        return None
    return ModelConfig.from_language_model(model)
