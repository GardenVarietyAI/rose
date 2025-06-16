"""Model registry for managing base and fine-tuned models."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from rose_server.config import ServiceConfig
from rose_server.model_registry import get_llm_models

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all LLM models."""

    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self._registry_path = Path(ServiceConfig.DATA_DIR) / "fine_tuned_models.json"
        self._load_base_models()
        self._load_fine_tuned_models()

    def _load_base_models(self):
        """Load base model configurations from settings."""
        self.models.update(get_llm_models())
        logger.info(f"Loaded {len(self.models)} base model configurations")

    def _load_fine_tuned_models(self):
        """Load fine-tuned models from registry file."""
        if not self._registry_path.exists():
            logger.debug("No fine-tuned models registry found")
            return
        try:
            with open(self._registry_path, "r") as f:
                registry = json.load(f)
            loaded = 0
            for model_id, model_info in registry.items():
                if Path(model_info["path"]).exists():
                    self.register_model(
                        model_id=model_id,
                        model_path=model_info["path"],
                        base_model=model_info.get("base_model", "qwen2.5-0.5b"),
                        persist=False,
                    )
                    loaded += 1
                else:
                    logger.warning(f"Model path not found for {model_id}: {model_info['path']}")
            logger.info(f"Loaded {loaded} fine-tuned models from registry")
        except Exception as e:
            logger.error(f"Error loading fine-tuned models registry: {e}")

    def register_model(
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
            persist: Whether to save fine-tuned models to registry file
        Returns:
            True if registration successful
        """
        try:
            if model_path:
                if not Path(model_path).exists():
                    logger.error(f"Model path does not exist: {model_path}")
                    return False
                base_config = self.models.get(base_model, {}).copy()
                base_config.update(
                    {"model_name": model_id, "model_path": model_path, "base_model": base_model, "is_fine_tuned": True}
                )
                if config:
                    base_config.update(config)
                self.models[model_id] = base_config
                if persist:
                    self._save_fine_tuned_registry()
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

    def unregister_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id not in self.models:
            return False
        is_fine_tuned = self.models[model_id].get("is_fine_tuned", False)
        del self.models[model_id]
        if is_fine_tuned:
            self._save_fine_tuned_registry()
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

    def _save_fine_tuned_registry(self):
        """Save fine-tuned models to registry file."""
        try:
            registry = {}
            for model_id, config in self.models.items():
                if config.get("is_fine_tuned", False):
                    registry[model_id] = {
                        "path": config["model_path"],
                        "base_model": config.get("base_model", "unknown"),
                        "created_at": config.get("created_at", 0),
                    }
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            logger.debug(f"Saved {len(registry)} fine-tuned models to registry")
        except Exception as e:
            logger.error(f"Failed to save fine-tuned models registry: {e}")
