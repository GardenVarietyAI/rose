"""Chat Completions API formatter for LLM events."""

import time
import uuid
from typing import Any, Dict, Optional

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from rose_server.events.event_types import (
    LLMEvent,
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallArgument,
    ToolCallCompleted,
    ToolCallStarted,
)


class ChatCompletionsFormatter:
    """Formats events for OpenAI Chat Completions API compatibility.

    This is the simplest formatter and proves our event system works
    with the existing OpenAI-compatible infrastructure.
    """

    def __init__(self) -> None:
        self.completion_id: Optional[str] = None
        self.created: Optional[int] = None
        self.model_name: Optional[str] = None
        self.request_seed: Optional[int] = None

    def set_request_seed(self, seed: Optional[int]) -> None:
        """Set the seed value from the request for fingerprint generation."""
        self.request_seed = seed

    def _get_base_chunk_dict(self) -> Dict[str, Any]:
        """Get base dictionary with common fields for all chunks."""
        return {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "system_fingerprint": f"fp_{self.model_name}" if self.request_seed is not None else None,
        }

    def format_event(self, event: LLMEvent) -> Optional[ChatCompletionChunk]:
        """Convert an LLM event to ChatCompletionChunk format."""

        if isinstance(event, ResponseStarted):
            self.completion_id = f"chatcmpl-{event.response_id}"
            self.created = int(event.timestamp)
            self.model_name = event.model_name

            return ChatCompletionChunk(
                **self._get_base_chunk_dict(),
                choices=[Choice(index=0, delta=ChoiceDelta(role="assistant"), finish_reason=None)],
            )
        elif isinstance(event, TokenGenerated):
            return ChatCompletionChunk(
                **self._get_base_chunk_dict(),
                choices=[Choice(index=0, delta=ChoiceDelta(content=event.token), finish_reason=None)],
            )
        elif isinstance(event, ToolCallStarted):
            return ChatCompletionChunk(
                **self._get_base_chunk_dict(),
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id=event.call_id,
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name=event.function_name,
                                        arguments="",
                                    ),
                                )
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
            )
        elif isinstance(event, ToolCallArgument):
            return ChatCompletionChunk(
                **self._get_base_chunk_dict(),
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(
                                        arguments=event.argument_delta,
                                    ),
                                )
                            ]
                        ),
                        finish_reason=None,
                    )
                ],
            )
        elif isinstance(event, ToolCallCompleted):
            return ChatCompletionChunk(
                **self._get_base_chunk_dict(),
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id=event.call_id,
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name=event.function_name,
                                        arguments=event.arguments,
                                    ),
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ],
            )
        elif isinstance(event, ResponseCompleted):
            return ChatCompletionChunk(
                id=self.completion_id or f"chatcmpl-{uuid.uuid4().hex[:16]}",
                object="chat.completion.chunk",
                created=self.created or int(time.time()),
                model=self.model_name or event.model_name,
                choices=[
                    Choice(
                        index=0,
                        delta=ChoiceDelta(),
                        finish_reason="stop"
                        if event.finish_reason in ["cancelled", "timeout"]
                        else event.finish_reason,
                    )
                ],
            )
        return None

    def format_complete_response(self, events: list[LLMEvent]) -> Dict[str, Any]:
        """Format a complete (non-streaming) response from all events."""
        start_event = next((e for e in events if isinstance(e, ResponseStarted)), None)
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        tool_events = [e for e in events if isinstance(e, ToolCallCompleted)]
        end_event = next((e for e in events if isinstance(e, ResponseCompleted)), None)
        content = "".join(e.token for e in token_events)
        message = {"role": "assistant"}
        message["content"] = content
        if tool_events:
            tool_calls = []
            for tool_event in tool_events:
                tool_calls.append(
                    {
                        "id": tool_event.call_id,
                        "type": "function",
                        "function": {"name": tool_event.function_name, "arguments": tool_event.arguments},
                    }
                )
            message["tool_calls"] = tool_calls

        # Build choice dict
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls"
            if tool_events
            else (
                "stop"
                if end_event and end_event.finish_reason in ["cancelled", "timeout"]
                else (end_event.finish_reason if end_event else "stop")
            ),
        }

        return {
            "id": f"chatcmpl-{start_event.response_id if start_event else uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(start_event.timestamp if start_event else time.time()),
            "model": start_event.model_name if start_event else "unknown",
            "system_fingerprint": f"fp_{start_event.model_name}" if self.request_seed is not None else None,
            "choices": [choice],
            "usage": {
                "prompt_tokens": start_event.input_tokens if start_event else 0,
                "completion_tokens": end_event.output_tokens if end_event else len(token_events),
                "total_tokens": (start_event.input_tokens if start_event else 0)
                + (end_event.output_tokens if end_event else len(token_events)),
            },
        }
