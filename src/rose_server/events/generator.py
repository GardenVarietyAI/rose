import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from rose_server._inference import (
    CompleteEvent,
    ErrorEvent,
    GenerationKwargs,
    InferenceServer,
    InputTokensCountedEvent,
    Message,
    TokenEvent,
)
from rose_server.config.settings import settings
from rose_server.events.event_types.generation import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.events.tool_processor import ToolProcessor
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.models import ModelConfig
from rose_server.tools import format_tools_for_prompt

logger = logging.getLogger(__name__)


class EventGenerator:
    def __init__(self, config: ModelConfig, inference_server: InferenceServer) -> None:
        self.config = config
        self.model_name = config.model_name
        self._srv = inference_server
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_inference)
        self._resolved_paths = self._resolve_model_paths(config.model_path)
        self._model_kind = config.kind or self._get_model_kind(config.model_id)

    @staticmethod
    def _resolve_model_paths(model_path: str) -> Dict[str, str]:
        path = Path(model_path)

        if path.is_dir():
            gguf_files = list(path.glob("*.gguf"))
            if gguf_files:
                gguf_file = str(gguf_files[0])
                tokenizer_file = str(path / "tokenizer.json")
                return {"model_path": gguf_file, "tokenizer_path": tokenizer_file}
            else:
                tokenizer_file = str(path / "tokenizer.json")
                return {"model_path": str(path), "tokenizer_path": tokenizer_file}
        elif path.suffix == ".gguf":
            return {"model_path": str(path), "tokenizer_path": str(path)}
        else:
            raise ValueError(f"Unsupported model path: {path}")

    @staticmethod
    def _get_model_kind(model_id: str) -> str:
        model_kind_map = {
            "Qwen--Qwen3-0.6B": "qwen3",
            "Qwen--Qwen3-1.7B": "qwen3",
            "Qwen--Qwen3-1.7B-Base": "qwen3",
            "Qwen--Qwen3-4B": "qwen3",
            "Qwen--Qwen3-0.6B-GGUF": "qwen3_gguf",
            "Qwen--Qwen3-4B-GGUF": "qwen3_gguf",
            "janhq--Jan-v1-4B-GGUF": "qwen3_gguf",
        }
        return model_kind_map.get(model_id, "qwen3")

    def _convert_rust_event(
        self,
        ev: Union[TokenEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent],
        response_id: str,
        input_tokens: int,
        completion_tokens: int,
        tool_processor: Optional[ToolProcessor],
    ) -> List[Union[TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted]]:
        match ev:
            case InputTokensCountedEvent():
                logger.debug(f"Input tokens counted: {ev.input_tokens}")
                return []

            case TokenEvent():
                token_event = TokenGenerated(
                    response_id=response_id,
                    model_name=self.model_name,
                    token=ev.token,
                    token_id=ev.token_id,
                    position=ev.position,
                )

                # Process tools if needed
                if tool_processor:
                    tool_events, modified_event = tool_processor.process_token(token_event)
                    events = list(tool_events)
                    if modified_event:
                        events.append(modified_event)
                    return events
                else:
                    return [token_event]

            case CompleteEvent():
                return [
                    ResponseCompleted(
                        response_id=response_id,
                        model_name=self.model_name,
                        input_tokens=ev.input_tokens,
                        output_tokens=ev.output_tokens,
                        total_tokens=ev.total_tokens,
                        finish_reason=ev.finish_reason,
                    )
                ]

            case ErrorEvent():
                logger.error(f"Error occurred during inference: {ev.error}")
                return [
                    ResponseCompleted(
                        response_id=response_id,
                        model_name=self.model_name,
                        input_tokens=input_tokens,
                        output_tokens=completion_tokens,
                        total_tokens=input_tokens + completion_tokens,
                        finish_reason="stop",
                    )
                ]

            case _:
                logger.warning(f"Unknown event type: {type(ev)}")
                return []

    async def generate_events(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: bool = False,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[str] = "auto",
        seed: Optional[int] = None,
        chain_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted],
        None,
    ]:
        """Generate events for a model run."""
        max_tokens = max_tokens or self.config.max_response_tokens
        temperature = temperature or self.config.temperature

        start_event = ResponseStarted(
            model_name=self.model_name,
            input_tokens=0,  # Passed in ResponseCompleted
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield start_event

        stream = self._stream_inference(
            messages=messages,
            response_id=start_event.response_id,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_tools=enable_tools,
            tools=tools,
            seed=seed,
            chain_id=chain_id,
        )
        async for event in stream:
            yield event

    async def _stream_inference(
        self,
        messages: List[ChatMessage],
        response_id: str,
        temperature: float,
        max_tokens: int,
        enable_tools: bool,
        tools: Optional[List[Any]],
        seed: Optional[int] = None,
        chain_id: Optional[str] = None,
    ) -> AsyncGenerator[Union[TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted], None]:
        prompt = format_tools_for_prompt(tools) if enable_tools and tools else None
        tool_processor = ToolProcessor(self.model_name) if enable_tools else None

        async with self._semaphore:
            generation_kwargs = GenerationKwargs(
                model_path=self._resolved_paths["model_path"],
                tokenizer_path=self._resolved_paths["tokenizer_path"],
                model_kind=self._model_kind,
                response_chain_id=chain_id,
                temperature=temperature,
                max_input_tokens=2048,
                max_output_tokens=max_tokens,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repetition_penalty or 1.1,
                repeat_last_n=64,
                stop=None,
                seed=seed,
                chat_template=None,
                enable_thinking=False,
            )

            rust_messages = [Message(role=msg.role, content=msg.content) for msg in messages]
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

            task = asyncio.ensure_future(self._srv.stream_direct(generation_kwargs, on_event, rust_messages, prompt))
            task.add_done_callback(_on_done)

            input_tokens = 0
            completion_tokens = 0

            try:
                while True:
                    ev = await q.get()

                    if isinstance(ev, InputTokensCountedEvent):
                        input_tokens = ev.input_tokens
                    elif isinstance(ev, TokenEvent):
                        completion_tokens += 1

                    events = self._convert_rust_event(ev, response_id, input_tokens, completion_tokens, tool_processor)
                    for event in events:
                        yield event

                    if isinstance(ev, (CompleteEvent, ErrorEvent)):
                        break
            finally:
                try:
                    await task
                except Exception:
                    pass
