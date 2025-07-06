"""Light orchestrator for inference requests."""

import asyncio
import gc
import logging
import uuid
from typing import Any, Dict

import torch
from fastapi import WebSocket

from rose_core.config.settings import settings
from rose_core.models import cleanup_model_memory
from rose_inference.generation.backends.hf_generator import generate_stream
from rose_inference.generation.runner import load_model, log_memory_status

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

            model_info = await load_model(model_name, model_config)

            try:
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
            finally:
                # ALWAYS cleanup the model after use since we're not caching
                logger.info(f"[{stream_id}] Cleaning up model {model_name}")
                log_memory_status("before cleanup")

                if model_info.get("model"):
                    cleanup_model_memory(model_info["model"])

                # Force deletion of references
                if "model" in model_info:
                    del model_info["model"]
                if "tokenizer" in model_info:
                    del model_info["tokenizer"]
                del model_info

                # Aggressive cleanup
                gc.collect()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    # MPS doesn't have empty_cache, but we can try to force cleanup
                    torch.mps.synchronize()

                log_memory_status("after cleanup")

        except Exception as e:
            logger.error(f"[{stream_id}] Inference error: {e}")
            error_msg = {"type": "error", "error": str(e)}
            try:
                await websocket.send_json(error_msg)
            except Exception:
                pass  # Connection might be closed

        finally:
            # Additional cleanup to ensure memory is released
            logger.info(f"[{stream_id}] Inference completed")
