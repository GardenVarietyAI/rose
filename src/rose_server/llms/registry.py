"""Model registry for managing base and fine-tuned models."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from rose_core.config.service import DATA_DIR
from rose_server.language_models.store import (
    get as get_language_model,
    list_all,
)
from rose_server.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Simple registry that wraps database operations for models."""

    def __init__(self) -> None:
        self._tokenizer_cache: Dict[str, AutoTokenizer] = {}

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

    # Tokenizer methods
    async def get_tokenizer(self, model_id: str) -> AutoTokenizer:
        """Get or load tokenizer for a model."""
        if model_id not in self._tokenizer_cache:
            # Get model config to find the actual model name
            config = await self.get_model_config(model_id)
            if config:
                model_name = config.get("model_name", model_id)
            else:
                model_name = model_id

            try:
                self._tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=True,
                )
                logger.debug(f"Loaded tokenizer for {model_id}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {model_id}: {e}")
                raise
        return self._tokenizer_cache[model_id]

    @lru_cache(maxsize=10000)
    def _count_tokens_cached(self, text: str, model_id: str) -> int:
        """Cached token counting (sync for performance)."""
        if not text:
            return 0
        # This is a limitation - we can't use async in cached method
        # For now, assume tokenizer is already loaded
        if model_id in self._tokenizer_cache:
            tokenizer = self._tokenizer_cache[model_id]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        return 0

    async def count_tokens(self, text: str, model_id: str = "qwen-coder") -> int:
        """Count tokens in text."""
        tokenizer = await self.get_tokenizer(model_id)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    async def count_messages(self, messages: List[ChatMessage], model_id: str = "qwen-coder") -> Dict[str, int]:
        """Count tokens in chat messages."""
        if not messages:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        tokenizer = await self.get_tokenizer(model_id)
        if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
            raise ValueError(f"Model {model_id} does not have a chat template")

        hf_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        formatted = tokenizer.apply_chat_template(hf_messages, tokenize=True, add_generation_prompt=False)
        total_tokens = len(formatted)
        return {"prompt_tokens": total_tokens, "completion_tokens": 0, "total_tokens": total_tokens}
