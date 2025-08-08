"""Simplified model generation with streaming."""

import asyncio
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import torch
from transformers.generation.streamers import TextIteratorStreamer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelGenerationParams:
    model: Any
    tokenizer: Any
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: str = ""
    generation_kwargs: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


def _set_random_seeds(seed: int) -> None:
    """Set random seeds for deterministic generation."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_messages(
    messages: Optional[List[Dict[str, Any]]], tokenizer: Any, prompt: Optional[str] = None
) -> Optional[str]:
    """Format messages and/or prompt into a single string, or return None if nothing to format."""
    if messages:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            formatted: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback formatting
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
            prompt_parts.append("Assistant:")
            formatted = "\n\n".join(prompt_parts)

        if prompt:
            return f"{formatted}\n\n{prompt}"

        return formatted

    return prompt


async def _prepare_inputs(formatted_prompt: str, tokenizer: Any, model: Any) -> Dict[str, Any]:
    """Tokenize and prepare inputs for model generation.

    Returns:
        Tuple of (inputs_dict, input_token_count)
    """
    inputs: Dict[str, Any] = tokenizer(formatted_prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    return inputs


async def stream(params: ModelGenerationParams, inference_timeout: int = 120) -> AsyncIterator[Dict[str, Any]]:
    """Stream generation events from a model.

    Yields:
        Dict events with types: input_tokens_counted, token, complete, error
    """
    generation_kwargs = params.generation_kwargs or {}
    model = params.model
    tokenizer = params.tokenizer

    # Handle seed for deterministic generation
    seed = generation_kwargs.pop("seed", None)
    if seed is not None:
        _set_random_seeds(seed)

    # Format input
    formatted_prompt = _format_messages(params.messages, tokenizer, params.prompt)
    if formatted_prompt is None:
        yield {"type": "complete", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return

    # Tokenize and prepare inputs
    inputs = await _prepare_inputs(formatted_prompt, tokenizer, model)
    input_token_count = inputs["input_ids"].shape[1]

    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    # Setup streaming
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None and eos_id is not None:
        pad_id = eos_id  # use eos when tokenizers have no pad

    generation_kwargs.pop("logprobs", False)  # not needed in this context

    # Build generation kwargs
    gen_kwargs = {
        **generation_kwargs,
        **inputs,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    async def stream_tokens() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, lambda: next(streamer, None))
            if token is None:
                break
            yield token

    # Start blocking generate in a worker thread under inference mode
    def _blocking_generate() -> Any:
        with torch.inference_mode():
            return params.model.generate(**gen_kwargs)

    generation_task = asyncio.create_task(asyncio.to_thread(_blocking_generate))

    # Stream tokens
    position = 0
    try:
        async for token in stream_tokens():
            if token:
                yield {"type": "token", "token": token, "position": position}
                position += 1
    except Exception:
        logger.exception("Generation error")
        yield {"type": "error", "error": "internal error"}
        generation_task.cancel()
        return
    finally:
        try:
            await asyncio.wait_for(generation_task, timeout=inference_timeout)
        except asyncio.CancelledError:
            logger.warning("Generation task was cancelled")
            generation_task.cancel()
            raise
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.error(f"Generation timeout after {inference_timeout}s")
                generation_task.cancel()
            else:
                logger.exception("Unexpected error in generation cleanup")
                generation_task.cancel()

    yield {
        "type": "complete",
        "input_tokens": input_token_count,
        "output_tokens": position,
        "total_tokens": input_token_count + position,
    }


def _process_logprob(logprob: float) -> float:
    """Convert -inf logprob to a large negative number for JSON compatibility."""
    if math.isinf(logprob) and logprob < 0:
        return -100.0
    return logprob


def _extract_top_logprobs(log_probs: torch.Tensor, top_k: int, tokenizer: Any) -> List[Dict[str, Any]]:
    """
    Extract top-k tokens with their log probabilities.

    Args:
        log_probs: Tensor of shape (vocab_size,) containing log probabilities.
        top_k: Number of top tokens to return.
        tokenizer: Tokenizer with a `.decode()` method.

    Returns:
        A list of dicts with "token", "logprob", and "bytes".
    """
    if top_k <= 0:
        return []

    top_k = min(top_k, log_probs.size(-1))
    top_log_probs, top_indices = torch.topk(log_probs, k=top_k)

    return [
        {
            "token": token_text,
            "logprob": _process_logprob(lp.item()),
            "bytes": list(token_text.encode("utf-8")),
        }
        for idx, lp in zip(top_indices, top_log_probs)
        for token_text in [tokenizer.decode([idx.item()], skip_special_tokens=False)]
    ]


async def with_logprobs(params: ModelGenerationParams, inference_timeout: int = 120) -> AsyncIterator[Dict[str, Any]]:
    """Generate with logprobs using output_scores=True.

    Non-streaming generation that returns logprobs for each token.
    """
    generation_kwargs = params.generation_kwargs or {}
    model = params.model
    tokenizer = params.tokenizer

    # Handle seed for deterministic generation
    seed = generation_kwargs.pop("seed", None)
    if seed is not None:
        _set_random_seeds(seed)

    # Format input
    formatted_prompt = _format_messages(params.messages, tokenizer, params.prompt)
    if formatted_prompt is None:
        yield {"type": "complete", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return

    # Tokenize and prepare inputs
    inputs = await _prepare_inputs(formatted_prompt, tokenizer, model)
    input_token_count = inputs["input_ids"].shape[1]

    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None and eos_id is not None:
        pad_id = eos_id  # use eos when tokenizers have no pad

    # Extract logprobs parameters (remove them from generation_kwargs)
    generation_kwargs.pop("logprobs", False)
    top_logprobs = generation_kwargs.pop("top_logprobs", 0)

    # Build generation kwargs with extra parameters for logprobs
    gen_kwargs = {
        **generation_kwargs,
        **inputs,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    try:
        # Generate all at once
        with torch.no_grad():
            outputs = await asyncio.to_thread(model.generate, **gen_kwargs)

        # Extract generated tokens (skip input tokens)
        generated_ids = outputs.sequences[0][input_token_count:]

        # Process each token with its score
        for position, (token_id, score) in enumerate(zip(generated_ids, outputs.scores)):
            # Convert token_id tensor to integer
            token_id_int = token_id.item()

            # Decode token
            token_text = tokenizer.decode(token_id_int, skip_special_tokens=False)

            # Convert logits to log probabilities
            log_probs = torch.nn.functional.log_softmax(score[0], dim=-1)

            # Get log probability of the selected token
            token_logprob = _process_logprob(log_probs[token_id_int].item())

            # Get top alternatives if requested
            top_tokens_data = _extract_top_logprobs(log_probs, top_logprobs, tokenizer)

            yield {
                "type": "token",
                "token": token_text,
                "position": position,
                "logprob": token_logprob,
                "top_logprobs": top_tokens_data,
            }

        output_token_count = len(generated_ids)
        yield {
            "type": "complete",
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count,
        }

    except Exception:
        logger.exception("Generation with logprobs error")
        yield {"type": "error", "error": "internal error"}
