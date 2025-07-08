"""Model registry for managing base and fine-tuned models."""

import logging
from typing import List, Optional

from rose_server.models.store import (
    get as get_language_model,
    list_all,
)
from rose_server.types.models import ModelConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple registry that wraps database operations for models."""

    def __init__(self) -> None:
        pass

    async def initialize(self) -> None:
        """Initialize the registry (no-op since we use database directly)."""
        logger.info("Model registry initialized")

    async def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model from the database."""
        model = await get_language_model(model_id)
        if not model:
            return None

        return ModelConfig.from_language_model(model)

    async def list_models(self) -> List[str]:
        """List all model IDs from database."""
        models = await list_all()
        return [m.id for m in models]
