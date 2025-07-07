import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from rose_server.events.event_types.generation import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.inference.client import InferenceClient
from rose_server.schemas.chat import ChatMessage
from rose_server.tools import StreamingXMLDetector

logger = logging.getLogger(__name__)


class EventGenerator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.model_name: str = config["model_name"]
        self._current_tools: Optional[List[Any]] = None

    async def generate_events(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: bool = False,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs: Any,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted],
        None,
    ]:
        """Main entrypoint: yield events for a model run."""
        # Store tools for later use in _stream_generation
        self._current_tools = tools

        max_new = max_tokens or self.config.get("max_response_tokens", 2048)
        temp = temperature or self.config.get("temperature", 0.7)

        # Token counting happens in inference layer
        start_event = ResponseStarted(
            model_name=self.model_name,
            input_tokens=0,  # Will be updated from inference layer
            max_tokens=max_new,
            temperature=temp,
        )
        yield start_event

        # Pass messages for proper formatting
        async for event in self._stream_generation(messages, start_event.response_id, max_new, temp, enable_tools):
            yield event

    async def _stream_generation(
        self,
        messages: List[ChatMessage],
        response_id: str,
        max_new: int,
        temperature: float,
        enable_tools: bool,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted], None
    ]:
        # Use WebSocket inference instead of local generation
        client = InferenceClient()

        # Convert ChatMessage objects to dicts for serialization
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Only prepare prompt if we have tools
        prompt = None
        if enable_tools and self._current_tools:
            # Only format the tool instructions
            from rose_server.tools import format_tools_for_prompt

            prompt = format_tools_for_prompt(self._current_tools)

        # Prepare generation kwargs for the worker
        generation_kwargs = {
            "max_new_tokens": max_new,
            "temperature": temperature,
            "top_p": self.config.get("top_p", 0.9),
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
            "length_penalty": self.config.get("length_penalty", 1.0),
        }

        detector = StreamingXMLDetector() if enable_tools else None

        async for event in client.stream_inference(
            model_name=self.model_name,
            model_config=self.config,
            prompt=prompt,  # Tool instructions only
            generation_kwargs=generation_kwargs,
            response_id=response_id,
            messages=messages_dict,  # Conversation history
        ):
            # Handle tool detection if needed
            if isinstance(event, TokenGenerated) and detector:
                tool_events, plain = self._handle_tool_streaming(event.token, detector, event.token_id)
                for tool_event in tool_events:
                    yield tool_event
                if plain:
                    # Update the token event with processed text
                    event.token = plain
                    yield event
            else:
                yield event

    def _handle_tool_streaming(
        self, token: str, detector: Any, total_tokens: int
    ) -> Tuple[List[Union[ToolCallStarted, ToolCallCompleted]], Optional[str]]:
        """Process token for tool calls, returns (events_list, plain_text)"""
        events: List[Union[ToolCallStarted, ToolCallCompleted]] = []
        plain, call = detector.process_token(token)
        if call:
            call_id = f"call_{uuid.uuid4().hex[:16]}"
            events.append(
                ToolCallStarted(
                    model_name=self.model_name,
                    function_name=call["tool"],
                    call_id=call_id,
                    arguments_so_far="",
                )
            )
            events.append(
                ToolCallCompleted(
                    model_name=self.model_name,
                    function_name=call["tool"],
                    call_id=call_id,
                    arguments=json.dumps(call["arguments"]),
                )
            )
        return events, plain

    def _response_completed_zero(self) -> ResponseCompleted:
        return ResponseCompleted(
            model_name=self.model_name,
            response_id=f"resp_{uuid.uuid4().hex[:16]}",
            total_tokens=0,
            finish_reason="stop",
            output_tokens=0,
            completion_time=0.0,
        )
