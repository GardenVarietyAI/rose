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
    TokenEvent,
    ToolCallArgumentEvent,
    ToolCallEvent,
    ToolCallStartEvent,
)
from rose_server.events.event_types import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallArgument,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.events.service import (
    build_rust_messages,
    create_event_queue,
    resolve_repetition_penalty,
    resolve_temperature,
)
from rose_server.models.qwen_configs import get_qwen_config
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.models import ModelConfig
from rose_server.tools.service import format_tools_for_system_prompt

logger = logging.getLogger(__name__)


class EventGenerator:
    def __init__(
        self, config: ModelConfig, inference_server: InferenceServer, max_concurrent_inference: int = 2
    ) -> None:
        self.config = config
        self.model_name = config.model_name
        self._srv = inference_server
        self._semaphore = asyncio.Semaphore(max_concurrent_inference)
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
            # For direct GGUF file paths, tokenizer must be in same directory
            tokenizer_file = path.parent / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(
                    f"No tokenizer.json found in {path.parent}. "
                    f"GGUF models require tokenizer.json in the same directory."
                )
            return {"model_path": str(path), "tokenizer_path": str(tokenizer_file)}
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
            "Qwen--Qwen3-1.7B-GGUF": "qwen3_gguf",
            "Qwen--Qwen3-4B-GGUF": "qwen3_gguf",
            "janhq--Jan-v1-4B-GGUF": "qwen3_gguf",
        }
        return model_kind_map.get(model_id, "qwen3")

    def _convert_rust_event(
        self,
        ev: Union[TokenEvent, ToolCallEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent],
        response_id: str,
        input_tokens: int,
        completion_tokens: int,
    ) -> List[Union[TokenGenerated, ResponseCompleted]]:
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
                return [token_event]

            case ToolCallStartEvent():
                start_event = ToolCallStarted(
                    response_id=response_id,
                    model_name=self.model_name,
                    call_id=ev.call_id,
                    function_name=ev.name,
                )
                return [start_event]

            case ToolCallArgumentEvent():
                arg_event = ToolCallArgument(
                    response_id=response_id,
                    model_name=self.model_name,
                    call_id=ev.call_id,
                    argument_delta=ev.argument_delta,
                )
                return [arg_event]

            case ToolCallEvent():
                complete_event = ToolCallCompleted(
                    response_id=response_id,
                    model_name=self.model_name,
                    call_id=ev.call_id,
                    function_name=ev.name,
                    arguments=ev.arguments,
                )
                return [complete_event]

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
        Union[ResponseStarted, TokenGenerated, ResponseCompleted],
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
    ) -> AsyncGenerator[Union[TokenGenerated, ResponseCompleted], None]:
        async with self._semaphore:
            qwen_config = get_qwen_config(self.config.model_id)
            actual_temperature = resolve_temperature(qwen_config, temperature, enable_tools, self.config.model_id)
            actual_repetition = resolve_repetition_penalty(self.config, qwen_config, enable_tools, self.config.model_id)

            generation_kwargs = GenerationKwargs(
                model_path=self._resolved_paths["model_path"],
                tokenizer_path=self._resolved_paths["tokenizer_path"],
                model_kind=self._model_kind,
                response_chain_id=chain_id,
                temperature=actual_temperature,
                max_input_tokens=qwen_config.max_context_length,
                max_output_tokens=min(max_tokens, qwen_config.max_response_tokens),
                top_p=self.config.top_p,
                repeat_penalty=actual_repetition,
                repeat_last_n=64,
                stop=None,
                seed=seed,
                chat_template=None,
                enable_thinking=False,
            )

            tool_prompt = format_tools_for_system_prompt(tools) if enable_tools and tools else ""
            if tool_prompt:
                logger.info(f"Tool prompt generated: {tool_prompt[:200]}...")

            rust_messages = build_rust_messages(messages, tool_prompt)
            q, push_event, on_done = create_event_queue()
            task = asyncio.ensure_future(self._srv.stream_direct(generation_kwargs, push_event, rust_messages, None))
            task.add_done_callback(on_done)

            input_tokens = 0
            completion_tokens = 0

            try:
                while True:
                    ev = await q.get()

                    if isinstance(ev, InputTokensCountedEvent):
                        input_tokens = ev.input_tokens
                    elif isinstance(ev, TokenEvent):
                        completion_tokens += 1

                    events = self._convert_rust_event(ev, response_id, input_tokens, completion_tokens)
                    for event in events:
                        yield event

                    if isinstance(ev, (CompleteEvent, ErrorEvent)):
                        break
            finally:
                try:
                    await task
                except Exception:
                    pass
