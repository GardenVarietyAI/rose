"""Simplified inference handling without queues."""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from rose_inference.cache import ModelCache
from rose_inference.generator import generate_stream
from rose_inference.loader import load_model

logger = logging.getLogger(__name__)


class InferenceHandler:
    """Handles model inference with caching."""

    def __init__(self) -> None:
        self.cache = ModelCache()
        self.request_count = 0

    async def run_inference(
        self,
        config: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Union[str, None] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run inference and yield events."""
        self.request_count += 1
        request_id = f"req-{self.request_count}"

        # Get both identifiers from config
        model_id = config.get("model_id")  # For caching (e.g., "ft:...")
        model_name = config.get("model_name")  # For loading (e.g., "Qwen/...")

        if not model_id or not model_name:
            error_msg = f"Missing required config: model_id={model_id}, model_name={model_name}"
            logger.error(f"[{request_id}] {error_msg}")
            yield {"type": "error", "error": error_msg}
            return

        logger.info(f"[{request_id}] Processing inference for model ID: {model_id} (name: {model_name})")

        try:
            # Check cache using model_id
            model_info = self.cache.get(model_id)

            # Load if not cached
            if model_info is None:
                logger.info(f"[{request_id}] Cache miss, loading model: {model_name}")
                # Load model using config
                model_info = await load_model(config)
                # Cache using model_id
                self.cache.set(model_id, model_info)

            # Generate
            async for event in generate_stream(
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                messages=messages,
                prompt=prompt,
                generation_kwargs=generation_kwargs,
                config=config,
            ):
                yield event

        except Exception as e:
            logger.error(f"[{request_id}] Inference error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    def evict_cache(self) -> Dict[str, Any]:
        """Evict cached models."""
        self.cache.evict()
        return {
            "status": "evicted",
            "message": "Model cache cleared",
            "cache_status": self.cache.get_status(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get handler status."""
        return {
            "status": "ok",
            "cache_status": self.cache.get_status(),
            "request_count": self.request_count,
        }
