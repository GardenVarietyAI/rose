"""Model registry for managing base and fine-tuned models."""

import logging
from typing import Optional

from rose_server.models.store import get as get_language_model
from rose_server.schemas.models import ModelConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple registry that wraps database operations for models."""

    def __init__(self) -> None:
        logger.info("Model registry initialized")

    async def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model from the database."""
        model = await get_language_model(model_id)
        if not model:
            return None

        return ModelConfig.from_language_model(model)
