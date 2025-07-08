"""Long-running inference worker."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from rose_core.config.settings import settings
from rose_inference.backends.hf_generator import generate_stream
from rose_inference.cache import ModelCache

logger = logging.getLogger(__name__)


class Runner:
    """Long-running worker that handles inference requests with model caching."""

    def __init__(self) -> None:
        self.model_cache = ModelCache()
        self.num_requests = 0
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=100)
        self.task = None

    async def start_worker(self) -> None:
        """Background worker that processes the request queue."""
        logger.info("Starting inference worker loop")

        while True:
            try:
                # Get next request from queue
                request = await self.queue.get()

                # Process the request
                async for event in self._process_inference(request):
                    # Send event through the response queue
                    await request["response_queue"].put(event)

                # Signal completion
                await request["response_queue"].put(None)

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                # Send error to response queue if possible
                if "response_queue" in locals() and request:
                    await request["response_queue"].put({"type": "error", "error": str(e)})
                    await request["response_queue"].put(None)

    async def _process_inference(self, request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a single inference request."""
        model_name = request["model_name"]
        model_config = request["model_config"]
        generation_kwargs = request["generation_kwargs"]
        messages = request.get("messages")
        prompt = request.get("prompt", "")
        stream_id = request.get("stream_id", f"req-{self.num_requests}")

        self.num_requests += 1

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

            # Stream events directly from generate_stream
            async for event in generate_stream(
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                prompt=prompt,
                messages=messages,
                generation_kwargs=generation_kwargs,
                stream_id=stream_id,
            ):
                yield event

            logger.info(f"[{stream_id}] Inference completed successfully")

        except Exception as e:
            logger.error(f"[{stream_id}] Inference error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    async def run_inference(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: str = "",
        stream_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Queue an inference request and yield events as they're processed."""
        # Create response queue for this request
        response_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        # Add request to the processing queue
        request = {
            "model_name": model_name,
            "model_config": model_config,
            "generation_kwargs": generation_kwargs,
            "messages": messages,
            "prompt": prompt,
            "stream_id": stream_id,
            "response_queue": response_queue,
        }

        await self.queue.put(request)
        logger.info(f"Queued inference request (queue size: {self.queue.qsize()})")

        # Yield events as they come from the worker
        while True:
            event = await response_queue.get()
            if event is None:  # Completion signal
                break
            yield event

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
                "num_requests": self.num_requests,
                "max_concurrent": settings.max_concurrent_inference,
                "queue_depth": self.queue.qsize(),
                "queue_max_size": self.queue.maxsize,
            },
        }
