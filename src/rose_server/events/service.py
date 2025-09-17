"""Composable units for event stream processing."""

import asyncio
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from rose_server._inference import (
    CompleteEvent,
    ErrorEvent,
    InputTokensCountedEvent,
    Message,
    TokenEvent,
    ToolCallEvent,
)
from rose_server.models.qwen_configs import QwenModelConfig, should_use_tool_config
from rose_server.schemas.chat import ChatMessage
from rose_server.schemas.models import ModelConfig

logger = logging.getLogger(__name__)


def create_event_queue() -> (
    Tuple[
        asyncio.Queue[Union[TokenEvent, ToolCallEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent]],
        Callable[[Union[TokenEvent, ToolCallEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent]], None],
        Callable[[asyncio.Future[Any]], None],
    ]
):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Union[TokenEvent, ToolCallEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent]] = (
        asyncio.Queue()
    )

    def push_event(ev: Union[TokenEvent, ToolCallEvent, CompleteEvent, InputTokensCountedEvent, ErrorEvent]) -> None:
        loop.call_soon_threadsafe(q.put_nowait, ev)

    def on_done(t: asyncio.Future[Any]) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            error_event = ErrorEvent()
            error_event.error = repr(exc)
            loop.call_soon_threadsafe(q.put_nowait, error_event)

    return q, push_event, on_done


def build_rust_messages(messages: List[ChatMessage], tool_prompt: Optional[str]) -> List[Message]:
    rust_messages = []
    system_updated = False

    for msg in messages:
        content = msg.content or ""

        # Inject tools into first system message
        if tool_prompt and msg.role == "system" and not system_updated:
            content = f"{content}\n\n{tool_prompt}"
            system_updated = True

        rust_messages.append(Message(role=msg.role, content=content, tool_call_id=msg.tool_call_id))

    # If no system message existed, prepend one with tools
    if tool_prompt and not system_updated:
        rust_messages.insert(0, Message(role="system", content=tool_prompt, tool_call_id=None))
        logger.info("Added system message with tools")

    return rust_messages


def resolve_temperature(
    qwen_config: QwenModelConfig,
    temperature: float,
    enable_tools: bool,
    model_id: str,
) -> float:
    if enable_tools and should_use_tool_config(model_id, enable_tools):
        return float(qwen_config.tool_temperature)
    return temperature


def resolve_repetition_penalty(
    config: ModelConfig,
    qwen_config: QwenModelConfig,
    enable_tools: bool,
    model_id: str,
) -> float:
    if enable_tools and should_use_tool_config(model_id, enable_tools):
        return float(qwen_config.tool_repetition_penalty)
    return float(config.repetition_penalty or qwen_config.repetition_penalty)
