"""Formatter for assistant runs events to OpenAI-compatible SSE format."""
import json
import time
import uuid
from typing import Any, Dict, Optional

from ..event_types.generation import (
    ResponseCompleted,
    ResponseStarted,
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)


class RunsFormatter:
    """Format events into OpenAI Assistants API SSE format.

    Converts standard generation events into the specific format
    expected by the OpenAI Assistants API for streaming runs.
    """

    def __init__(self, run_id: str, thread_id: str, assistant_id: str, message_id: Optional[str] = None):
        """Initialize formatter with run context.

        Args:
            run_id: The ID of the run
            thread_id: The ID of the thread
            assistant_id: The ID of the assistant
            message_id: Optional message ID (created if not provided)
        """
        self.run_id = run_id
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.message_id = message_id or f"msg_{uuid.uuid4().hex[:16]}"
        self.accumulated_content = ""
        self.tool_calls = []

    def format_event(self, event: Any) -> Optional[str]:
        """Format a single event into SSE format.

        Returns None if the event shouldn't be sent to the client.
        """
        if isinstance(event, ResponseStarted):
            return self._format_sse(
                "thread.message.created",
                {
                    "id": self.message_id,
                    "object": "thread.message",
                    "created_at": int(time.time()),
                    "thread_id": self.thread_id,
                    "role": "assistant",
                    "content": [{"type": "text", "text": {"value": "", "annotations": []}}],
                    "assistant_id": self.assistant_id,
                    "run_id": self.run_id,
                    "file_ids": [],
                    "metadata": {},
                }
            )
        elif isinstance(event, TokenGenerated):
            self.accumulated_content += event.token
            return self._format_sse(
                "thread.message.delta",
                {
                    "id": self.message_id,
                    "object": "thread.message.delta",
                    "delta": {
                        "content": [{
                            "index": 0,
                            "type": "text",
                            "text": {"value": event.token}
                        }]
                    },
                }
            )
        elif isinstance(event, ToolCallStarted):
            self.tool_calls.append({
                "id": event.call_id,
                "type": "function",
                "function": {
                    "name": event.function_name,
                    "arguments": ""
                }
            })
            return None
        elif isinstance(event, ToolCallCompleted):
            for tc in self.tool_calls:
                if tc["id"] == event.call_id:
                    tc["function"]["arguments"] = event.arguments
                    break
            return None
        elif isinstance(event, ResponseCompleted):
            content = [{"type": "text", "text": {"value": self.accumulated_content, "annotations": []}}]
            if self.tool_calls:
                content.append({
                    "type": "tool_calls",
                    "tool_calls": self.tool_calls
                })
            return self._format_sse(
                "thread.message.completed",
                {
                    "id": self.message_id,
                    "object": "thread.message.completed",
                    "created_at": int(time.time()),
                    "thread_id": self.thread_id,
                    "role": "assistant",
                    "content": content,
                    "assistant_id": self.assistant_id,
                    "run_id": self.run_id,
                    "file_ids": [],
                    "metadata": {},
                }
            )
        return None

    def _format_sse(self, event_type: str, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"