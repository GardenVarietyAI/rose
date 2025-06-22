"""Model registry for managing base and fine-tuned models."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rose_core.config.service import DATA_DIR, LLM_MODELS

from .store import (
    create as create_language_model,
    delete as delete_language_model,
    list_fine_tuned,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all LLM models."""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_base_models()

    def _load_base_models(self):
        """Load base model configurations from settings."""
        self.models.update(LLM_MODELS.copy())
        logger.info(f"Loaded {len(self.models)} base model configurations")

    async def initialize(self):
        """Initialize the registry with fine-tuned models from database."""
        await self._load_fine_tuned_models()

    async def _load_fine_tuned_models(self):
        """Load fine-tuned models from database."""
        try:
            fine_tuned_models = await list_fine_tuned()
            loaded = 0
            for model in fine_tuned_models:
                if model.path:
                    model_path = Path(DATA_DIR) / model.path
                    if model_path.exists():
                        # Get base model config and update it
                        base_config = self.models.get(model.base_model, {}).copy()
                        base_config.update(
                            {
                                "model_name": model.id,
                                "name": model.name,
                                "model_path": str(model_path),
                                "base_model": model.base_model,
                                "is_fine_tuned": True,
                                "created_at": model.created_at,
                            }
                        )
                        self.models[model.id] = base_config
                        loaded += 1
                    else:
                        logger.warning(f"Model path not found for {model.id}: {model_path}")
            logger.info(f"Loaded {loaded} fine-tuned models from database")
        except Exception as e:
            logger.error(f"Error loading fine-tuned models from database: {e}")

    async def register_model(
        self,
        model_id: str,
        model_path: Optional[str] = None,
        base_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> bool:
        """Register a model in the registry.
        Args:
            model_id: Unique identifier for the model
            model_path: Path to fine-tuned model directory (for fine-tuned models)
            base_model: Base model name (for fine-tuned models)
            config: Full configuration (for base models or overrides)
            persist: Whether to save fine-tuned models to database
        Returns:
            True if registration successful
        """
        try:
            if model_path:
                if not Path(model_path).exists():
                    logger.error(f"Model path does not exist: {model_path}")
                    return False

                # Store in database if persist is True
                if persist:
                    # Get relative path for storage
                    relative_path = Path(model_path).relative_to(DATA_DIR)
                    hf_model_name = self.models.get(base_model, {}).get("hf_model_name") if base_model else None

                    await create_language_model(
                        id=model_id,
                        path=str(relative_path),
                        base_model=base_model,
                        hf_model_name=hf_model_name,
                    )

                # Update in-memory registry
                base_config = self.models.get(base_model, {}).copy() if base_model else {}
                base_config.update(
                    {
                        "model_name": model_id,
                        "model_path": model_path,
                        "base_model": base_model,
                        "is_fine_tuned": True,
                        "created_at": None,  # Will be set from database
                    }
                )
                if config:
                    base_config.update(config)
                self.models[model_id] = base_config
            else:
                if not config:
                    logger.error(f"No config provided for base model {model_id}")
                    return False
                self.models[model_id] = config
            logger.info(f"Registered model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error registering model {model_id}: {e}")
            return False

    async def unregister_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id not in self.models:
            return False
        is_fine_tuned = self.models[model_id].get("is_fine_tuned", False)

        # Delete from database if it's a fine-tuned model
        if is_fine_tuned:
            await delete_language_model(model_id)

        # Remove from in-memory registry
        del self.models[model_id]
        logger.info(f"Unregistered model: {model_id}")
        return True

    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a model."""
        return self.models.get(model_id)

    def list_models(self) -> list[str]:
        """List all registered model IDs."""
        return list(self.models.keys())

    def list_fine_tuned_models(self) -> list[str]:
        """List only fine-tuned model IDs."""
        return [model_id for model_id, config in self.models.items() if config.get("is_fine_tuned", False)]

    async def reload_fine_tuned_models(self):
        """Reload fine-tuned models from database."""
        # Remove existing fine-tuned models from memory
        fine_tuned_ids = [model_id for model_id, config in self.models.items() if config.get("is_fine_tuned", False)]
        for model_id in fine_tuned_ids:
            del self.models[model_id]

        # Reload from database
        await self._load_fine_tuned_models()
        logger.info("Reloaded fine-tuned models from database")
