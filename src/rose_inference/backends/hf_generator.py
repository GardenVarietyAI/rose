"""HuggingFace model generation logic."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from transformers import TextIteratorStreamer  # type: ignore[attr-defined]

from rose_core.config.settings import settings

logger = logging.getLogger(__name__)


def _format_messages_fallback(messages: List[Dict[str, Any]]) -> str:
    """Fallback formatting when tokenizer doesn't have a chat template."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append("Assistant:")  # Add generation prompt
    return "\n\n".join(prompt_parts)


async def generate_stream(
    model: Any,
    tokenizer: Any,
    prompt: str,
    messages: Optional[List[Dict[str, Any]]],
    generation_kwargs: Dict[str, Any],
    stream_id: str,
) -> AsyncIterator[Dict[str, Any]]:
    """Run model generation and yield events.

    Yields events with types:
    - input_tokens_counted: After tokenization
    - token: For each generated token
    - complete: When generation finishes
    - error: If an error occurs
    """
    # Handle different input combinations
    if messages:
        # Format messages using chat template or fallback
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            logger.info(f"[{stream_id}] Using chat template for {len(messages)} messages")
            formatted_messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            logger.info(f"[{stream_id}] No chat template, using fallback formatting")
            formatted_messages = _format_messages_fallback(messages)

        # If we have both messages and prompt, prepend messages as context
        if prompt:
            prompt = f"{formatted_messages}\n\n{prompt}"
        else:
            prompt = formatted_messages

    # Tokenize input
    logger.info(f"[{stream_id}] Tokenizing prompt (length: {len(prompt)})")
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get input token count
    input_token_count = inputs["input_ids"].shape[1]
    logger.info(f"[{stream_id}] Input tokens: {input_token_count}")

    # Yield input token count
    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    # Create streamer for token-by-token output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Add streamer to generation kwargs
    generation_kwargs = {
        **generation_kwargs,
        **inputs,
        "streamer": streamer,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # Start generation in background thread (required for streaming)
    generation_task = asyncio.create_task(asyncio.to_thread(model.generate, **generation_kwargs))

    # Stream tokens
    position = 0
    try:
        # Create async wrapper for the blocking streamer
        async def stream_tokens() -> AsyncIterator[str]:
            loop = asyncio.get_event_loop()
            while True:
                token = await loop.run_in_executor(None, lambda: next(streamer, None))
                if token is None:
                    break
                yield token

        async for token in stream_tokens():
            if token:  # Skip empty tokens
                yield {"type": "token", "token": token, "position": position}
                position += 1

        logger.info(f"[{stream_id}] Generated {position} tokens")

    except Exception as e:
        logger.error(f"[{stream_id}] Error during generation: {e}")
        yield {"type": "error", "error": str(e)}
        generation_task.cancel()
        raise
    finally:
        # Wait for generation to complete
        try:
            await asyncio.wait_for(generation_task, timeout=settings.inference_timeout)
        except asyncio.TimeoutError:
            logger.error(f"[{stream_id}] Generation timeout after {settings.inference_timeout}s")
            generation_task.cancel()

    # Yield completion with token counts
    yield {
        "type": "complete",
        "input_tokens": input_token_count,
        "output_tokens": position,
        "total_tokens": input_token_count + position,
    }
