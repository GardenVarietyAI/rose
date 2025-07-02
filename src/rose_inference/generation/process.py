"""Light orchestrator for inference requests."""

import asyncio
import logging
import uuid
from typing import Any, Dict

from fastapi import WebSocket

from rose_core.config.settings import settings
from rose_core.models import cleanup_model_memory
from rose_inference.generation.backends.hf_generator import generate_stream
from rose_inference.generation.cache import load_model

logger = logging.getLogger(__name__)

# Semaphore for controlling concurrent inferences (default: 1 = sequential)
inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inference)


async def process_inference_request(websocket: WebSocket, request_data: Dict[str, Any]) -> None:
    """Process a single inference request - orchestrator."""
    stream_id = str(uuid.uuid4())[:8]  # Short ID for request tracking

    async with inference_semaphore:  # Simple concurrency control
        try:
            logger.info(f"[{stream_id}] Starting inference request")

            # Extract request data
            model_name = request_data["model_name"]
            model_config = request_data["config"]
            generation_kwargs = request_data["generation_kwargs"]
            messages = request_data.get("messages")  # Optional messages for chat formatting
            prompt = request_data.get("prompt", "")  # Optional pre-formatted prompt

            # Check for empty input
            if not messages and not prompt:
                logger.info(f"[{stream_id}] Empty input received, returning empty response")
                await websocket.send_json({"type": "complete", "total_tokens": 0})
                return

            # Load model (cached)
            model_info = await load_model(model_name, model_config)

            # Run generation and stream results
            token_counts = await generate_stream(
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                prompt=prompt,
                messages=messages,
                generation_kwargs=generation_kwargs,
                websocket=websocket,
                stream_id=stream_id,
            )

            # Send completion with token counts
            await websocket.send_json(
                {
                    "type": "complete",
                    "input_tokens": token_counts["input_tokens"],
                    "output_tokens": token_counts["output_tokens"],
                    "total_tokens": token_counts["input_tokens"] + token_counts["output_tokens"],
                }
            )

        except Exception as e:
            logger.error(f"[{stream_id}] Inference error: {e}")
            error_msg = {"type": "error", "error": str(e)}
            try:
                await websocket.send_json(error_msg)
            except Exception:
                pass  # Connection might be closed

        finally:
            # Cleanup memory after each inference
            cleanup_model_memory()
            logger.info(f"[{stream_id}] Inference completed")
