import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from rose_server._inference import CompleteEvent, ErrorEvent, InferenceServer, InputTokensCountedEvent, TokenEvent
from rose_server.config.settings import settings
from rose_server.events.event_types.generation import (
    ResponseCompleted,
    TokenGenerated,
)

logger = logging.getLogger(__name__)


# Model ID to model kind mapping
MODEL_KIND_MAP = {
    "Qwen--Qwen3-0.6B": "qwen3",
    "Qwen--Qwen3-1.7B": "qwen3",
    "Qwen--Qwen3-1.7B-Base": "qwen3",
    "Qwen--Qwen3-4B": "qwen3",
    "Qwen--Qwen3-0.6B-GGUF": "qwen3_gguf",
    "janhq--Jan-v1-4B-GGUF": "qwen3_gguf",
    "Qwen--Qwen2.5-0.5B-Instruct": "qwen2",
    "Qwen--Qwen2.5-1.5B-Instruct": "qwen2",
    "Qwen--Qwen2.5-Coder-1.5B-Instruct": "qwen2",
}


def resolve_model_paths(model_path: str) -> Dict[str, str]:
    """Resolve model and tokenizer paths."""
    from pathlib import Path

    path = Path(model_path)

    if path.is_dir():
        gguf_files = list(path.glob("*.gguf"))
        if gguf_files:
            gguf_file = str(gguf_files[0])
            # tokenizer should be in same directory as GGUF file
            tokenizer_file = str(path / "tokenizer.json")
            return {"model_path": gguf_file, "tokenizer_path": tokenizer_file}
        else:
            # pass directory path to Rust for regular models,
            tokenizer_file = str(path / "tokenizer.json")
            return {"model_path": str(path), "tokenizer_path": tokenizer_file}
    elif path.suffix == ".gguf":
        return {"model_path": str(path), "tokenizer_path": str(path)}
    else:
        raise ValueError(f"Unsupported model path: {path}")


class InferenceClient:
    """Minimal local client that bridges Rust events to your event types."""

    def __init__(self, device: str = "auto") -> None:
        self._srv = InferenceServer(device)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_inference)

    async def stream_inference(
        self,
        *,
        model_name: str,
        model_config: Dict[str, Any],
        prompt: Optional[str],
        generation_kwargs: Dict[str, Any],
        response_id: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[Union[TokenGenerated, ResponseCompleted], None]:
        """Call the inference service."""
        async with self._semaphore:
            # Resolve paths and get model kind
            resolved_paths = resolve_model_paths(model_config.get("model_path", ""))
            model_kind = MODEL_KIND_MAP.get(model_config.get("model_id", ""), "qwen2")

            req: Dict[str, Any] = {
                "generation_kwargs": {
                    "model_path": resolved_paths["model_path"],
                    "tokenizer_path": resolved_paths["tokenizer_path"],
                    "model_kind": model_kind,
                    "temperature": generation_kwargs.get("temperature", 0.7),
                    "max_input_tokens": generation_kwargs.get("max_input_tokens", 8192),
                    "max_output_tokens": generation_kwargs.get("max_output_tokens", 1024),
                    "stop": None,
                    "logprobs": generation_kwargs.get("logprobs"),
                    "top_logprobs": generation_kwargs.get("top_logprobs"),
                    "chat_template": generation_kwargs.get("chat_template"),
                    "enable_thinking": generation_kwargs.get("enable_thinking", False),
                }
            }

            if prompt is not None:
                req["prompt"] = prompt
            elif messages is not None:
                req["messages"] = messages

            loop = asyncio.get_running_loop()
            q: asyncio.Queue[Union[TokenEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent]] = asyncio.Queue()

            def on_event(ev: Union[TokenEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent]) -> None:
                loop.call_soon_threadsafe(q.put_nowait, ev)

            def _on_done(t: asyncio.Future[Any]) -> None:
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None:
                    error_event = ErrorEvent()
                    error_event.error = repr(exc)
                    loop.call_soon_threadsafe(q.put_nowait, error_event)

            task = asyncio.ensure_future(self._srv.stream(req, on_event))
            task.add_done_callback(_on_done)

            input_tokens = 0
            completion_tokens = 0

            try:
                while True:
                    ev = await q.get()

                    if isinstance(ev, InputTokensCountedEvent):
                        input_tokens = ev.input_tokens
                        logger.debug(f"Input tokens counted: {input_tokens}")

                    elif isinstance(ev, TokenEvent):
                        completion_tokens += 1
                        yield TokenGenerated(
                            response_id=response_id,
                            model_name=model_name,
                            token=ev.token,
                            token_id=ev.token_id,
                            position=ev.position,
                            logprob=ev.logprob,
                            top_logprobs=ev.top_logprobs,
                        )

                    elif isinstance(ev, CompleteEvent):
                        yield ResponseCompleted(
                            response_id=response_id,
                            model_name=model_name,
                            input_tokens=ev.input_tokens,
                            output_tokens=ev.output_tokens,
                            total_tokens=ev.total_tokens,
                            finish_reason=ev.finish_reason,
                        )
                        break

                    elif isinstance(ev, ErrorEvent):
                        error_msg = ev.error
                        logger.error(f"Error occurred during inference: {error_msg}")
                        yield ResponseCompleted(
                            response_id=response_id,
                            model_name=model_name,
                            input_tokens=input_tokens,
                            output_tokens=completion_tokens,
                            total_tokens=input_tokens + completion_tokens,
                            finish_reason="stop",
                        )
                        break
            finally:
                try:
                    await task
                except Exception:
                    pass

    def flush_model(self) -> None:
        """Free cached model memory"""
        self._srv.flush_model()
