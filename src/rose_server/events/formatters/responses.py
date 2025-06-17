"""Responses API formatter for LLM events."""

import json
import time
import uuid
from typing import Any, Dict, Optional, Sequence

from ...schemas.responses import (
    ResponsesContentItem,
    ResponsesOutputItem,
    ResponsesResponse,
    ResponsesUsage,
)
from ...tools import parse_xml_tool_call
from ..event_types import (
    LLMEvent,
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)


class ResponsesFormatter:
    """Formats events for Responses API"""

    def __init__(self) -> None:
        self.response_id: Optional[str] = None
        self.created_at: Optional[int] = None
        self.model_name: Optional[str] = None
        self.accumulated_content: str = ""  # Track content for streaming

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
            return None
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

        # Accumulate content
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
        # Use accumulated content from streaming
        output_items = self._build_output_items_from_content(self.accumulated_content)

        # Convert to dicts for JSON serialization in streaming
        output_dicts = [item.model_dump() for item in output_items]

        return {
            "type": "response.completed",
            "sequence_number": 999,
            "response": {
                "id": self.response_id or f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": self.created_at or int(time.time()),
                "model": self.model_name or event.model_name,
                "status": "completed",
                "output": output_dicts,
                "usage": self._build_usage(event),
            },
        }

    def _build_output_items_from_content(self, content: str) -> list[ResponsesOutputItem]:
        """Build output items from content string."""
        output_items: list[ResponsesOutputItem] = []

        if not content:
            return output_items

        # Check for tool calls
        tool_call, cleaned_text = parse_xml_tool_call(content)
        if tool_call:
            function_item = ResponsesOutputItem(
                id=f"call_{uuid.uuid4().hex[:16]}",
                type="function_call",
                status="completed",
                role="assistant",
                name=tool_call["tool"],
                arguments=json.dumps(tool_call["arguments"]),
            )
            output_items.append(function_item)
            content = cleaned_text

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
        return {
            "input_tokens": getattr(event, "input_tokens", 0),
            "output_tokens": getattr(event, "output_tokens", 0),
            "total_tokens": getattr(event, "input_tokens", 0) + getattr(event, "output_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    def format_complete_response(self, events: Sequence[LLMEvent]) -> Dict[str, Any]:
        """Format a complete (non-streaming) response from all events."""
        start_event = next((e for e in events if isinstance(e, ResponseStarted)), None)
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        end_event = next((e for e in events if isinstance(e, ResponseCompleted)), None)
        content = "".join(e.token for e in token_events)

        # Use the helper method to build output items
        output_items = self._build_output_items_from_content(content)

        usage = ResponsesUsage(
            input_tokens=start_event.input_tokens if start_event else 0,
            output_tokens=end_event.output_tokens if end_event and end_event.output_tokens else len(token_events),
            total_tokens=(start_event.input_tokens if start_event else 0)
            + (end_event.output_tokens if end_event and end_event.output_tokens else len(token_events)),
        )

        response = ResponsesResponse(
            id=start_event.response_id if start_event else f"resp_{uuid.uuid4().hex}",
            created_at=int(start_event.timestamp if start_event else time.time()),
            model=start_event.model_name if start_event else "unknown",
            status="completed",
            output=output_items,
            usage=usage,
        )

        return response.model_dump()
