"""Dependency injection for language models."""

from typing import Annotated

from fastapi import Depends, Request

from .registry import ModelRegistry


def get_model_registry(request: Request) -> ModelRegistry:
    """Get the model registry from app state."""
    return request.app.state.model_registry


ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
