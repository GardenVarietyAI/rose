"""Long-running inference worker."""

import logging
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from rose_core.config.settings import settings
from rose_core.models import get_tokenizer, load_hf_model
from rose_inference.cache import ModelCache
from rose_inference.generation.backends.hf_generator import generate_stream

logger = logging.getLogger(__name__)


async def load_model(model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load a model and tokenizer for inference."""
    logger.info(f"Loading model: {model_name}")

    loader: Callable[..., Any] = model_config.get("loader", load_hf_model)

    # Use model_path if available (for custom/fine-tuned models), otherwise use model_name
    model_id = model_config.get("model_path") or model_config.get("model_name", model_name)

    # Load the model based on loader type
    if loader.__name__ == "load_peft_model":
        # PEFT models require model_path
        model = loader(
            model_id=model_id,
            model_path=model_config.get("model_path"),
            torch_dtype=model_config.get("torch_dtype"),
        )
    else:
        # Regular HF models only need model_id
        model = loader(
            model_id=model_id,
            torch_dtype=model_config.get("torch_dtype"),
        )

    # Load tokenizer
    tokenizer = get_tokenizer(model_id)

    # Return model info
    return {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "config": model_config,
        "device": str(model.device) if hasattr(model, "device") else "cpu",
        "dtype": str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "unknown",
    }


class InferenceWorker:
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


# Global worker instance
_worker: Optional[InferenceWorker] = None


def get_worker() -> InferenceWorker:
    """Get or create the global worker instance."""
    global _worker
    if _worker is None:
        _worker = InferenceWorker()
    return _worker
