"""Long-running inference worker."""

import logging
from typing import Any, AsyncGenerator, Dict, Optional

from rose_core.config.settings import settings
from rose_inference.backends.hf_generator import generate_stream
from rose_inference.cache import ModelCache

logger = logging.getLogger(__name__)


class Runner:
    """Long-running worker that handles inference requests with model caching."""

    def __init__(self) -> None:
        self.model_cache = ModelCache()
        self.request_count = 0

    async def run_inference(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        messages: Optional[list] = None,
        prompt: str = "",
        stream_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run inference and yield events."""
        if not stream_id:
            stream_id = f"req-{self.request_count}"
        self.request_count += 1

        try:
            logger.info(f"[{stream_id}] Processing inference request for model: {model_name}")

            # Check for empty input
            if not messages and not prompt:
                logger.info(f"[{stream_id}] Empty input received")
                yield {
                    "type": "complete",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
                return

            # Get or load model
            model_info = await self.model_cache.get_or_load_model(model_name, model_config)

            # Create event collector
            events = []

            class EventCollector:
                """Collects events from generate_stream."""

                async def send_json(self, data: Dict[str, Any]) -> None:
                    events.append(data)

            collector = EventCollector()

            # Run generation
            logger.info(f"[{stream_id}] Starting generation")
            token_counts = await generate_stream(
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                prompt=prompt,
                messages=messages,
                generation_kwargs=generation_kwargs,
                websocket=collector,  # Pass collector instead of websocket
                stream_id=stream_id,
            )

            # Yield all collected events
            for event in events:
                yield event

            # Yield completion
            yield {
                "type": "complete",
                "input_tokens": token_counts["input_tokens"],
                "output_tokens": token_counts["output_tokens"],
                "total_tokens": token_counts["input_tokens"] + token_counts["output_tokens"],
            }

            logger.info(f"[{stream_id}] Inference completed successfully")

        except Exception as e:
            logger.error(f"[{stream_id}] Inference error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    def evict_models(self) -> Dict[str, Any]:
        """Evict all cached models."""
        logger.info("Evicting all models")
        self.model_cache.evict()
        return {
            "status": "evicted",
            "message": "Model cache cleared",
            "cache_status": self.model_cache.get_status(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get worker and cache status."""
        return {
            "status": "ok",
            "cache_status": self.model_cache.get_status(),
            "worker_status": {
                "request_count": self.request_count,
                "max_concurrent": settings.max_concurrent_inference,
            },
        }


# Runner class is now just a plain class - no singleton needed
# Instance will be managed by FastAPI's app.state
