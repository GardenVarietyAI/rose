"""Embedding model manager following the TransformersManager pattern."""

import logging
import os
import threading
from typing import Any, Dict

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from rose_core.config.models import get_embedding_models

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manager for SentenceTransformer embedding models only."""

    def __init__(self):
        self.models: Dict[str, SentenceTransformer] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load_configs()

    def _load_configs(self):
        """Load embedding model configurations from app settings."""
        self.model_configs.update(get_embedding_models())
        logger.info(f"Loaded {len(self.model_configs)} embedding model configurations")

    def register_model(self, model_id: str, model_path: str) -> bool:
        """Register a custom embedding model."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return False
            self.model_configs[model_id] = {
                "model_name": model_path,
                "dimensions": 384,
                "description": f"Custom model: {model_id}",
                "format": "HuggingFace",
            }
            logger.info(f"Registered embedding model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register embedding model {model_id}: {e}")
            return False

    def get_model(self, model_id: str) -> SentenceTransformer:
        """Get or load an embedding model."""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not found in available models: {list(self.model_configs.keys())}")
        with self._lock:
            if model_id not in self.models:
                config = self.model_configs[model_id]
                model_name = config["model_name"]
                logger.info(f"Loading embedding model: {model_id} ({model_name})")
                self.models[model_id] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded embedding model: {model_id}")
            return self.models[model_id]

    def get_tokenizer(self, model_id: str) -> AutoTokenizer:
        """Get or load tokenizer for an embedding model."""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not found in available models: {list(self.model_configs.keys())}")
        with self._lock:
            if model_id not in self.tokenizers:
                config = self.model_configs[model_id]
                model_name = config["model_name"]
                logger.info(f"Loading tokenizer for embedding model: {model_id} ({model_name})")
                self.tokenizers[model_id] = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, use_fast=True
                )
                logger.info(f"Successfully loaded tokenizer for: {model_id}")
            return self.tokenizers[model_id]

    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model to free memory."""
        with self._lock:
            success = False
            if model_id in self.models:
                del self.models[model_id]
                success = True
                logger.info(f"Unloaded embedding model: {model_id}")
            if model_id in self.tokenizers:
                del self.tokenizers[model_id]
                logger.info(f"Unloaded tokenizer for: {model_id}")
            if success:
                import gc

                gc.collect()
            return success

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available embedding models."""
        return self.model_configs.copy()

    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of which models are currently loaded."""
        return {
            model_id: {"model_loaded": model_id in self.models, "tokenizer_loaded": model_id in self.tokenizers}
            for model_id in self.model_configs
        }

    def cleanup(self):
        """Clean up all loaded models."""
        with self._lock:
            self.models.clear()
            self.tokenizers.clear()
            import gc

            gc.collect()
            logger.info("Cleaned up all embedding models")
