"""Light orchestrator for inference requests."""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict

from rose_core.config.service import MAX_CONCURRENT_INFERENCE, MAX_PROMPT_LENGTH
from rose_core.models import cleanup_model_memory

from .backends.hf_generator import generate_stream
from .cache import load_model

logger = logging.getLogger(__name__)

# Semaphore for controlling concurrent inferences (default: 1 = sequential)
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


async def process_inference_request(websocket, request_data: Dict[str, Any]) -> None:
    """Process a single inference request - orchestrator."""
    stream_id = str(uuid.uuid4())[:8]  # Short ID for request tracking

    async with inference_semaphore:  # Simple concurrency control
        try:
            logger.info(f"[{stream_id}] Starting inference request")

            # Extract request data
            model_name = request_data["model_name"]
            model_config = request_data["config"]
            prompt = request_data["prompt"]
            generation_kwargs = request_data["generation_kwargs"]
            messages = request_data.get("messages")  # Optional messages for chat formatting

            # Validate prompt length before doing any work
            if len(prompt) > MAX_PROMPT_LENGTH:
                logger.warning(f"[{stream_id}] Prompt too long: {len(prompt)} > {MAX_PROMPT_LENGTH}")
                await websocket.send(
                    json.dumps(
                        {"type": "error", "error": f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters"}
                    )
                )
                return

            # Load model (cached)
            model_info = await load_model(model_name, model_config)

            # Run generation and stream results
            total_tokens = await generate_stream(
                model=model_info["model"],
                tokenizer=model_info["tokenizer"],
                prompt=prompt,
                messages=messages,
                generation_kwargs=generation_kwargs,
                websocket=websocket,
                stream_id=stream_id,
            )

            # Send completion
            await websocket.send(json.dumps({"type": "complete", "total_tokens": total_tokens}))

        except Exception as e:
            logger.error(f"[{stream_id}] Inference error: {e}")
            error_msg = {"type": "error", "error": str(e)}
            try:
                await websocket.send(json.dumps(error_msg))
            except Exception:
                pass  # Connection might be closed

        finally:
            # Cleanup memory after each inference
            cleanup_model_memory()
            logger.info(f"[{stream_id}] Inference completed")
