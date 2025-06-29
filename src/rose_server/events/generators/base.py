import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from rose_server.events import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.llms.llm import LLM
from rose_server.llms.websocket_inference import InferenceClient
from rose_server.schemas.chat import ChatMessage
from rose_server.tools import StreamingXMLDetector

logger = logging.getLogger(__name__)


class BaseEventGenerator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.model_name: str = llm.model_name
        self.config: Dict[str, Any] = llm.config
        self.last_prompt: Optional[str] = None

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
        """Main entrypoint: yield events for an LLM run."""
        prompt = await self.prepare_prompt(messages, enable_tools=enable_tools, tools=tools)
        self.last_prompt = prompt  # Store for WebSocket inference

        max_new = max_tokens or self.config.get("max_response_tokens", 512)
        temp = temperature or self.config.get("temperature", 0.7)

        # Estimate input tokens (rough approximation without tokenizer)
        input_tokens = len(prompt.split()) * 2  # Rough estimate

        start_event = ResponseStarted(
            model_name=self.model_name,
            input_tokens=input_tokens,
            max_tokens=max_new,
            temperature=temp,
        )
        yield start_event

        # Pass empty inputs since we're using WebSocket
        async for event in self._stream_generation({}, start_event.response_id, max_new, temp, enable_tools):
            yield event

    async def prepare_prompt(
        self, messages: List[ChatMessage], enable_tools: bool = False, tools: Optional[List[Any]] = None
    ) -> str:
        """Override to customize prompt construction."""
        formatted: Any = self.llm.format_messages(messages)
        return str(formatted)

    async def _stream_generation(
        self,
        inputs: Dict[str, Any],
        response_id: str,
        max_new: int,
        temperature: float,
        enable_tools: bool,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted], None
    ]:
        # Use WebSocket inference instead of local generation
        client = InferenceClient()

        # Convert inputs back to prompt (since we already formatted it)
        prompt = self.last_prompt  # We'll need to store this in generate_events

        # Prepare generation kwargs for the worker
        generation_kwargs = {
            "max_new_tokens": max_new,
            "temperature": temperature,
            "top_p": self.config.get("top_p", 0.9),
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
            "length_penalty": self.config.get("length_penalty", 1.0),
        }

        # Get model config
        model_config = {
            "model_name": self.llm.model_name,
            "model_path": self.llm.model_path,
            "torch_dtype": self.config.get("torch_dtype"),
        }

        detector = StreamingXMLDetector() if enable_tools else None

        async for event in client.stream_inference(
            model_name=self.model_name,
            model_config=model_config,
            prompt=prompt,
            generation_kwargs=generation_kwargs,
            response_id=response_id,
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
