"""Model cache for keeping models hot in memory."""

import logging
import time
from typing import Any, Dict, Optional

from rose_core.models import cleanup_model_memory
from rose_inference.generation.runner import load_model

logger = logging.getLogger(__name__)


class ModelCache:
    """Simple model cache that keeps the last used model hot."""

    def __init__(self) -> None:
        self.current_model: Optional[Dict[str, Any]] = None
        self.current_model_name: Optional[str] = None
        self.last_used: float = time.time()
        self.total_inferences: int = 0

    async def get_or_load_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get model from cache or load it."""
        # Check if we have the same model cached
        if self.current_model_name == model_name:
            logger.info(f"Using cached model: {model_name} (inference #{self.total_inferences + 1})")
            self.last_used = time.time()
            self.total_inferences += 1
            return self.current_model

        # Different model requested - evict old one if exists
        if self.current_model:
            logger.info(f"Evicting model: {self.current_model_name} (served {self.total_inferences} inferences)")
            self.evict()

        # Load new model
        logger.info(f"Loading new model: {model_name}")
        self.current_model = await load_model(model_name, model_config)
        self.current_model_name = model_name
        self.last_used = time.time()
        self.total_inferences = 1

        return self.current_model

    def evict(self) -> None:
        """Evict the cached model to free memory."""
        if self.current_model:
            logger.info(f"Evicting model: {self.current_model_name}")
            # Clean up model
            if "model" in self.current_model:
                del self.current_model["model"]
            if "tokenizer" in self.current_model:
                del self.current_model["tokenizer"]
            self.current_model = None
            self.current_model_name = None
            self.total_inferences = 0

            # Force memory cleanup
            cleanup_model_memory()
            logger.info("Model evicted and memory cleaned")

    def get_status(self) -> Dict[str, Any]:
        """Get cache status."""
        return {
            "cached_model": self.current_model_name,
            "last_used": self.last_used,
            "total_inferences": self.total_inferences,
            "cache_age_seconds": time.time() - self.last_used if self.current_model else 0,
        }
