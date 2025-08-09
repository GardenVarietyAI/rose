"""Simplified model generation with streaming."""

import asyncio
import logging
import math
import queue
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.utils.import_utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelGenerationParams:
    model: Any
    tokenizer: Any
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: str = ""
    generation_kwargs: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


def _model_device(model: Any) -> torch.device:
    try:
        device: torch.device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return device


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
    inputs: Dict[str, Any] = tokenizer(formatted_prompt, return_tensors="pt")
    dev = _model_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    return inputs


def _pick_attn_impl() -> Optional[str]:
    if torch.cuda.is_available() and is_flash_attn_2_available():  # type: ignore[no-untyped-call]
        return "flash_attention_2"
    # On CPU/MPS keep it eager
    # Some stacks passing the arg on MPS confuses things, so return None and  don't include it.
    return None


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
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.2)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None and eos_id is not None:
        pad_id = eos_id  # use eos when tokenizers have no pad

    generation_kwargs.pop("logprobs", None)  # not needed in this context

    # Build generation kwargs
    gen_kwargs = {
        **generation_kwargs,
        **inputs,
        "max_new_tokens": generation_kwargs.get("max_new_tokens", 256),
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "streamer": streamer,
        "use_cache": True,
    }

    attn_impl = _pick_attn_impl()
    if attn_impl is not None:
        gen_kwargs["attn_implementation"] = attn_impl

    # Start blocking generate in a worker thread under inference mode
    def _blocking_generate() -> Any:
        with torch.inference_mode():
            return model.generate(**gen_kwargs)

    generation_task = asyncio.create_task(asyncio.to_thread(_blocking_generate))

    _EOS = object()

    def _poll_streamer() -> Any:
        try:
            return next(streamer)  # may raise StopIteration
        except queue.Empty:
            return None  # no token yet
        except StopIteration:
            return _EOS  # end of stream

    async def stream_tokens() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        while True:
            res = await loop.run_in_executor(None, _poll_streamer)
            if res is _EOS:
                break  # clean EOS
            if res is None:
                # no token yet, if generation is finished then loop once more to catch EOS
                if generation_task.done():
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0)
                continue
            yield res

    # Stream tokens
    position = 0
    try:
        async for token in stream_tokens():
            yield {"type": "token", "token": token, "position": position}
            position += 1
    except Exception:
        logger.exception("Error during token generation")
        yield {"type": "error", "error": "internal error"}
        if not generation_task.done():
            generation_task.cancel()
        # don't re-raise
    finally:
        # wait for the background thread to finish (or give up)
        try:
            await asyncio.wait_for(generation_task, timeout=inference_timeout)
        except asyncio.TimeoutError:
            generation_task.cancel()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unexpected error in generation cleanup")

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
    """Non-streaming generation that yields per-token logprobs (decoder-only)."""
    generation_kwargs = params.generation_kwargs or {}
    model = params.model
    tokenizer = params.tokenizer

    # Deterministic seeding (optional)
    seed = generation_kwargs.pop("seed", None)
    if seed is not None:
        _set_random_seeds(seed)

    # Format input
    formatted_prompt = _format_messages(params.messages, tokenizer, params.prompt)
    if formatted_prompt is None:
        yield {"type": "complete", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return

    # Tokenize and move to model device
    inputs = await _prepare_inputs(formatted_prompt, tokenizer, model)
    input_token_count = inputs["input_ids"].shape[1]
    yield {"type": "input_tokens_counted", "input_tokens": input_token_count}

    # IDs
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # Logprobs controls (strip from user kwargs)
    generation_kwargs.pop("logprobs", None)
    top_logprobs = int(generation_kwargs.pop("top_logprobs", 0))

    # Build generate kwargs
    gen_kwargs: Dict[str, Any] = {
        **generation_kwargs,
        **inputs,
        "do_sample": generation_kwargs.get("temperature", 1.0) > 0,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "output_scores": True,
        "return_dict_in_generate": True,
        "use_cache": True,
    }

    # Omit attn_implementation on MPS/CPU
    attn_impl = _pick_attn_impl()
    if attn_impl is not None:
        gen_kwargs["attn_implementation"] = attn_impl

    # run sync generate in a worker thread
    def _gen_sync() -> GenerateDecoderOnlyOutput:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        return cast(GenerateDecoderOnlyOutput, out)

    try:
        outputs = await asyncio.wait_for(asyncio.to_thread(_gen_sync), timeout=inference_timeout)

        # Slice out new tokens
        generated_ids: torch.Tensor = outputs.sequences[0][input_token_count:]

        scores_tuple: Optional[Tuple[torch.FloatTensor, ...]] = cast(
            Optional[Tuple[torch.FloatTensor, ...]], outputs.scores
        )
        if scores_tuple is None:
            # Nothing to yield, return a well-formed completion
            yield {
                "type": "complete",
                "input_tokens": input_token_count,
                "output_tokens": 0,
                "total_tokens": input_token_count,
            }
            return

        # Iterate by index so types line up (zip would complain if scores is Optional)
        for position in range(len(scores_tuple)):
            score: torch.FloatTensor = scores_tuple[position]  # (batch, vocab)
            token_id = generated_ids[position]  # scalar tensor
            token_id_int = int(token_id.item())
            token_text = tokenizer.decode([token_id_int], skip_special_tokens=False)

            log_probs = F.log_softmax(score[0], dim=-1)  # batch=1
            token_logprob = _process_logprob(float(log_probs[token_id_int].item()))
            top_tokens_data = _extract_top_logprobs(log_probs, top_logprobs, tokenizer)

            yield {
                "type": "token",
                "token": token_text,
                "position": position,
                "logprob": token_logprob,
                "top_logprobs": top_tokens_data,
            }

        yield {
            "type": "complete",
            "input_tokens": input_token_count,
            "output_tokens": len(scores_tuple),
            "total_tokens": input_token_count + len(scores_tuple),
        }

    except asyncio.TimeoutError:
        logger.error(f"Generation timeout after {inference_timeout}s")
        yield {"type": "error", "error": "timeout"}
    except Exception:
        logger.exception("Generation with logprobs error")
        yield {"type": "error", "error": "internal error"}
