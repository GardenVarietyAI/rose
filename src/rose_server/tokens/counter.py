"""HuggingFace-native tokenizer service for exact token counting."""
import logging
from functools import lru_cache
from typing import Dict, List
from transformers import AutoTokenizer
from rose_server.schemas.chat import ChatMessage
logger = logging.getLogger(__name__)

class TokenizerService:
    """HuggingFace tokenizer service with OpenAI-compatible responses."""

    def __init__(self, model_registry=None):
        self._tokenizer_cache: Dict[str, AutoTokenizer] = {}
        self._model_registry = model_registry

    def _get_tokenizer(self, model_id: str) -> AutoTokenizer:
        """Get cached tokenizer for model."""
        if model_id not in self._tokenizer_cache:
            model_path = model_id
            if self._model_registry:
                config = self._model_registry.get_model_config(model_id)
                if config:
                    model_path = config.get("model_path") or config.get("model_name", model_id)
            try:
                self._tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True,
                )
                logger.debug(f"Loaded new tokenizer for {model_id}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {model_id}: {e}")
                raise
        return self._tokenizer_cache[model_id]
    @lru_cache(maxsize=10000)

    def _count_tokens_cached(self, text: str, model_id: str) -> int:
        """Internal cached token counting function."""
        if not text:
            return 0
        tokenizer = self._get_tokenizer(model_id)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def count_tokens(self, text: str, model_id: str = "qwen-coder") -> int:
        """Count tokens using exact HuggingFace tokenizer."""
        return self._count_tokens_cached(text, model_id)

    def count_messages(self, messages: List[ChatMessage], model_id: str = "qwen-coder") -> Dict[str, int]:
        """Count tokens in chat messages with OpenAI-compatible format."""
        if not messages:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            tokenizer = self._get_tokenizer(model_id)
            if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
                raise ValueError(f"Model {model_id} does not have a chat template")
            hf_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            formatted = tokenizer.apply_chat_template(hf_messages, tokenize=True, add_generation_prompt=False)
            total_tokens = len(formatted)
            return {"prompt_tokens": total_tokens, "completion_tokens": 0, "total_tokens": total_tokens}
        except Exception as e:
            logger.error(f"Error counting message tokens for {model_id}: {e}")
            raise

    def tokenize(self, text: str, model_id: str = "qwen-coder") -> List[int]:
        """Tokenize text into token IDs."""
        if not text:
            return []
        tokenizer = self._get_tokenizer(model_id)
        return tokenizer.encode(text, add_special_tokens=False)

    def estimate_tokens_fast(self, text: str, model_id: str = "qwen-coder") -> int:
        """Fast token estimation for streaming performance."""
        return max(1, len(text) // 4)

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        cache_info = self._count_tokens_cached.cache_info()
        return {
            "lru_hits": cache_info.hits,
            "lru_misses": cache_info.misses,
            "lru_current_size": cache_info.currsize,
            "lru_max_size": cache_info.maxsize,
            "tokenizer_count": len(self._tokenizer_cache),
        }