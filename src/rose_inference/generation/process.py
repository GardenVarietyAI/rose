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
            generation_kwargs = request_data["generation_kwargs"]
            messages = request_data.get("messages")  # Optional messages for chat formatting
            prompt = request_data.get("prompt", "")  # Optional pre-formatted prompt

            # Check for empty input
            if not messages and not prompt:
                logger.info(f"[{stream_id}] Empty input received, returning empty response")
                await websocket.send(json.dumps({"type": "complete", "total_tokens": 0}))
                return

            # Validate input length before doing any work
            # For messages, do a rough estimate
            if messages:
                estimated_length = sum(len(str(msg)) for msg in messages)
                if estimated_length > MAX_PROMPT_LENGTH:
                    logger.warning(f"[{stream_id}] Messages too long: ~{estimated_length} > {MAX_PROMPT_LENGTH}")
                    error_msg = f"Input exceeds maximum length of {MAX_PROMPT_LENGTH} characters"
                    await websocket.send(json.dumps({"type": "error", "error": error_msg}))
                    return
            elif len(prompt) > MAX_PROMPT_LENGTH:
                logger.warning(f"[{stream_id}] Prompt too long: {len(prompt)} > {MAX_PROMPT_LENGTH}")
                error_msg = f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters"
                await websocket.send(json.dumps({"type": "error", "error": error_msg}))
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
            await websocket.send(
                json.dumps(
                    {
                        "type": "complete",
                        "input_tokens": token_counts["input_tokens"],
                        "output_tokens": token_counts["output_tokens"],
                        "total_tokens": token_counts["input_tokens"] + token_counts["output_tokens"],
                    }
                )
            )

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
