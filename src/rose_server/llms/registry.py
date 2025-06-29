"""Model registry for managing base and fine-tuned models."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rose_core.config.service import DATA_DIR
from rose_server.models.store import (
    get as get_language_model,
    list_all,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple registry that wraps database operations for models."""

    def __init__(self) -> None:
        pass

    async def initialize(self) -> None:
        """Initialize the registry (no-op since we use database directly)."""
        logger.info("Model registry initialized")

    async def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a model from the database."""
        model = await get_language_model(model_id)
        if not model:
            return None

        # Start with model fields
        config: dict[str, Any] = model.model_dump(
            include={
                "model_name",
                "model_type",
                "temperature",
                "top_p",
                "memory_gb",
                "timeout",
            }
        )

        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}

        # Add computed fields
        if model.is_fine_tuned and model.path:
            config["model_path"] = str(Path(DATA_DIR) / model.path)
            config["base_model"] = model.parent
            config["is_fine_tuned"] = True

        if model.get_lora_modules():
            config["lora_target_modules"] = model.get_lora_modules()

        return config

    async def list_models(self) -> List[str]:
        """List all model IDs from database."""
        models = await list_all()
        return [m.id for m in models]
