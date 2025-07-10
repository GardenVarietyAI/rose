import logging
from typing import Any, AsyncGenerator, List, Optional, Union

from rose_server.events.event_types.generation import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.events.tool_processor import ToolProcessor
from rose_server.inference.client import InferenceClient
from rose_server.schemas.chat import ChatMessage
from rose_server.tools import format_tools_for_prompt
from rose_server.types.models import ModelConfig

logger = logging.getLogger(__name__)


class EventGenerator:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self.client = InferenceClient()

    async def generate_events(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: bool = False,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[str] = "auto",
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[
        Union[ResponseStarted, TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted],
        None,
    ]:
        """Generate events for a model run."""
        # Use provided values or defaults
        max_tokens = max_tokens or self.config.max_response_tokens
        temperature = temperature or self.config.temperature

        # Emit start event
        start_event = ResponseStarted(
            model_name=self.model_name,
            input_tokens=0,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield start_event

        # Stream from inference
        async for event in self._stream_inference(
            messages=messages,
            response_id=start_event.response_id,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_tools=enable_tools,
            tools=tools,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        ):
            yield event

    async def _stream_inference(
        self,
        messages: List[ChatMessage],
        response_id: str,
        temperature: float,
        max_tokens: int,
        enable_tools: bool,
        tools: Optional[List[Any]],
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ) -> AsyncGenerator[Union[TokenGenerated, ToolCallStarted, ToolCallCompleted, ResponseCompleted], None]:
        """Stream events from the inference service."""
        # Prepare messages
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Prepare tool prompt if needed
        prompt = format_tools_for_prompt(tools) if enable_tools and tools else None

        # Build generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "length_penalty": self.config.length_penalty,
        }

        # Add logprobs parameters if provided
        if logprobs is not None:
            generation_kwargs["logprobs"] = logprobs
            generation_kwargs["top_logprobs"] = top_logprobs or 0

        # Create tool processor if needed
        tool_processor = ToolProcessor(self.model_name) if enable_tools else None

        # Stream from inference
        model_config_dict = self.config.model_dump()
        logger.info(f"Streaming inference for {self.model_name} with config: {model_config_dict}")
        async for event in self.client.stream_inference(
            model_name=self.model_name,
            model_config=model_config_dict,
            prompt=prompt,
            generation_kwargs=generation_kwargs,
            response_id=response_id,
            messages=messages_dict,
        ):
            # Process tools if needed
            if isinstance(event, TokenGenerated) and tool_processor:
                tool_events, modified_event = tool_processor.process_token(event)

                # Yield tool events first
                for tool_event in tool_events:
                    yield tool_event

                # Yield modified token event if any
                if modified_event:
                    yield modified_event
            else:
                yield event
