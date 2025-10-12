"""Responses API formatter for LLM events."""

import time
import uuid
from typing import Any, Dict, Optional, Sequence

from rose_server.events.event_types import (
    LLMEvent,
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.schemas.responses import (
    ResponsesContentItem,
    ResponsesOutputItem,
    ResponsesResponse,
    ResponsesUsage,
)


class ResponsesFormatter:
    """Formats events for Responses API"""

    def __init__(self) -> None:
        self.response_id: Optional[str] = None
        self.created_at: Optional[int] = None
        self.model_name: Optional[str] = None
        self.accumulated_content: str = ""
        self.tool_calls: list[Dict[str, Any]] = []

    def format_event(self, event: LLMEvent) -> Optional[Dict[str, Any]]:
        """Convert an LLM event to Responses API streaming format."""
        if isinstance(event, ResponseStarted):
            self._handle_response_started(event)
            return None
        elif isinstance(event, TokenGenerated):
            return self._handle_token_generated(event)
        elif isinstance(event, ToolCallStarted):
            return None
        elif isinstance(event, ToolCallCompleted):
            self.tool_calls.append(
                {
                    "id": event.call_id,
                    "name": event.function_name,
                    "arguments": event.arguments,
                }
            )
            return {
                "type": "response.output_item.done",
                "output_index": len(self.tool_calls) - 1,
                "item": {
                    "id": event.call_id,
                    "type": "function_call",
                    "role": "assistant",
                    "name": event.function_name,
                    "arguments": event.arguments,
                    "call_id": event.call_id,
                },
            }
        elif isinstance(event, ResponseCompleted):
            return self._handle_response_completed(event)
        return None

    def _handle_response_started(self, event: ResponseStarted) -> None:
        self.response_id = event.response_id
        self.created_at = int(event.timestamp)
        self.model_name = event.model_name

    def _handle_token_generated(self, event: TokenGenerated) -> Dict[str, Any]:
        if not self.response_id:
            self.response_id = f"resp_{uuid.uuid4().hex}"
            self.created_at = int(time.time())
            self.model_name = event.model_name

        self.accumulated_content += event.token

        return {
            "type": "response.output_text.delta",
            "sequence_number": event.position,
            "item_id": f"item_{self.response_id}",
            "output_index": 0,
            "content_index": 0,
            "delta": event.token,
        }

    def _handle_response_completed(self, event: ResponseCompleted) -> Dict[str, Any]:
        """Handle ResponseCompleted event."""
        output_items = []

        # Format function calls as separate items
        for tool_call in self.tool_calls:
            function_item = {
                "id": tool_call["id"],
                "call_id": tool_call["id"],
                "type": "function_call",
                "role": "assistant",
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
            }
            output_items.append(function_item)

        # Add text content as a message item if any
        if self.accumulated_content and self.accumulated_content.strip():
            content_item = ResponsesContentItem(type="output_text", text=self.accumulated_content)
            output_item = ResponsesOutputItem(
                id=f"msg_{uuid.uuid4().hex}",
                type="message",
                status="completed",
                role="assistant",
                content=[content_item],
            )
            output_items.append(output_item.model_dump())

        return {
            "type": "response.completed",
            "sequence_number": 999,
            "response": {
                "id": self.response_id or f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": self.created_at or int(time.time()),
                "model": self.model_name or event.model_name,
                "status": "completed",
                "output": output_items,
                "usage": self._build_usage(event),
            },
        }

    def _build_output_items_from_content(self, content: str) -> list[ResponsesOutputItem]:
        """Build output items from content string."""
        output_items: list[ResponsesOutputItem] = []

        if not content:
            return output_items

        # Add text content if any
        if content and content.strip():
            content_item = ResponsesContentItem(type="output_text", text=content)
            output_item = ResponsesOutputItem(
                id=f"msg_{uuid.uuid4().hex}",
                type="message",
                status="completed",
                role="assistant",
                content=[content_item],
            )
            output_items.append(output_item)

        return output_items

    def _build_usage(self, event: ResponseCompleted) -> Dict[str, Any]:
        """Build usage stats from event."""
        input_tokens = event.input_tokens
        output_tokens = event.output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    def format_complete_response(self, events: Sequence[LLMEvent]) -> ResponsesResponse:
        """Format a complete (non-streaming) response from all events."""
        start_event = next((e for e in events if isinstance(e, ResponseStarted)), None)
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        tool_events = [e for e in events if isinstance(e, ToolCallCompleted)]
        end_event = next((e for e in events if isinstance(e, ResponseCompleted)), None)
        content = "".join(e.token for e in token_events)

        output_items = []

        # Add tool calls if any
        for tool_event in tool_events:
            function_item = {
                "id": tool_event.call_id,
                "call_id": tool_event.call_id,
                "type": "function_call",
                "role": "assistant",
                "name": tool_event.function_name,
                "arguments": tool_event.arguments,
            }
            output_items.append(function_item)

        # Add text content as a message item if any
        if content and content.strip():
            content_item = ResponsesContentItem(type="output_text", text=content)
            output_item = ResponsesOutputItem(
                id=f"msg_{uuid.uuid4().hex}",
                type="message",
                status="completed",
                role="assistant",
                content=[content_item],
            )
            output_items.append(output_item.model_dump())

        # Get input tokens and output tokens from ResponseCompleted event
        input_tokens = end_event.input_tokens if end_event else 0
        output_tokens = (
            end_event.output_tokens if end_event and end_event.output_tokens is not None else len(token_events)
        )

        usage = ResponsesUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        response = ResponsesResponse(
            id=start_event.response_id if start_event else f"resp_{uuid.uuid4().hex}",
            created_at=int(start_event.timestamp if start_event else time.time()),
            model=start_event.model_name if start_event else "unknown",
            status="completed",
            output=output_items,
            usage=usage,
        )

        return response
