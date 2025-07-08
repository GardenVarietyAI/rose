"""Model cache for keeping models hot in memory."""

import logging
import time
from typing import Any, Dict, Optional

from rose_inference.model_loader import cleanup_model_memory

logger = logging.getLogger(__name__)


class ModelCache:
    """Simple model cache that keeps the last used model hot."""

    def __init__(self) -> None:
        self.current_model: Optional[Dict[str, Any]] = None
        self.current_model_name: Optional[str] = None
        self.last_used: float = time.time()
        self.total_inferences: int = 0

    def get(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model from cache if it exists."""
        if self.current_model_name == model_name and self.current_model is not None:
            logger.info(f"Cache hit for model: {model_name} (inference #{self.total_inferences + 1})")
            self.last_used = time.time()
            self.total_inferences += 1
            return self.current_model
        return None

    def set(self, model_name: str, model_info: Dict[str, Any]) -> None:
        """Set a model in the cache, evicting any existing model."""
        # Evict old model if different
        if self.current_model_name and self.current_model_name != model_name:
            logger.info(f"Evicting model: {self.current_model_name} to make room for {model_name}")
            self.evict()

        # Cache new model
        logger.info(f"Caching model: {model_name}")
        self.current_model = model_info
        self.current_model_name = model_name
        self.last_used = time.time()
        self.total_inferences = 1

    def evict(self) -> None:
        """Evict the cached model to free memory."""
        if self.current_model:
            logger.info(f"Evicting model: {self.current_model_name}")

            # Pass model to cleanup for proper PEFT handling
            if "model" in self.current_model:
                cleanup_model_memory(self.current_model["model"])

            # Clean up remaining references
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
            "cache_type": "single_model",
            "cache_description": "Keeps one model hot in memory. Loading a different model evicts the current one.",
            "cached_model": self.current_model_name,
            "last_used": self.last_used,
            "total_inferences": self.total_inferences,
            "cache_age_seconds": time.time() - self.last_used if self.current_model else 0,
            "is_loaded": self.current_model is not None,
        }
