"""Responses API formatter for LLM events."""

import json
import time
import uuid
from typing import Any, Dict, Optional

from rose_server.tools import parse_xml_tool_call

from .. import LLMEvent, ResponseCompleted, ResponseStarted, TokenGenerated, ToolCallCompleted, ToolCallStarted


class ResponsesFormatter:
    """Formats events for OpenAI Responses API compatibility.

    This formatter converts LLM events to the specific format expected
    by the /v1/responses endpoint, which differs from chat completions.
    """

    def __init__(self):
        self.response_id: Optional[str] = None
        self.created_at: Optional[int] = None
        self.model_name: Optional[str] = None

    def format_event(self, event: LLMEvent) -> Optional[Dict[str, Any]]:
        """Convert an LLM event to Responses API streaming format."""

        if isinstance(event, ResponseStarted):
            self.response_id = event.response_id
            self.created_at = int(event.timestamp)
            self.model_name = event.model_name
            return None
        elif isinstance(event, TokenGenerated):
            if not self.response_id:
                self.response_id = f"resp_{uuid.uuid4().hex}"
                self.created_at = int(time.time())
                self.model_name = event.model_name
            return {
                "type": "response.output_text.delta",
                "sequence_number": len(event.token),
                "item_id": f"item_{self.response_id}",
                "output_index": 0,
                "content_index": 0,
                "delta": event.token,
            }
        elif isinstance(event, ToolCallStarted):
            return None
        elif isinstance(event, ToolCallCompleted):
            return None
        elif isinstance(event, ResponseCompleted):
            output_items = []
            if hasattr(event, "final_text") and event.final_text:
                tool_call, cleaned_text = parse_xml_tool_call(event.final_text)
                if tool_call:
                    output_items.append(
                        {
                            "type": "function_call",
                            "id": f"call_{uuid.uuid4().hex[:16]}",
                            "name": tool_call["tool"],
                            "arguments": json.dumps(tool_call["arguments"]),
                        }
                    )
                    final_text = cleaned_text
                else:
                    final_text = event.final_text
            else:
                final_text = getattr(event, "content", "")
            if final_text and final_text.strip():
                output_items.append(
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "role": "assistant",
                        "content": [{"type": "text", "text": final_text}],
                    }
                )
            return {
                "type": "response.completed",
                "sequence_number": 999,
                "response": {
                    "id": self.response_id or f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "created": self.created_at or int(time.time()),
                    "model": self.model_name or event.model_name,
                    "status": "completed",
                    "output": output_items,
                    "usage": {
                        "input_tokens": event.input_tokens if hasattr(event, "input_tokens") else 0,
                        "output_tokens": event.output_tokens if hasattr(event, "output_tokens") else 0,
                        "total_tokens": (getattr(event, "input_tokens", 0) + getattr(event, "output_tokens", 0)),
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            }
        return None

    def format_complete_response(self, events: list[LLMEvent]) -> Dict[str, Any]:
        """Format a complete (non-streaming) response from all events."""
        start_event = next((e for e in events if isinstance(e, ResponseStarted)), None)
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        end_event = next((e for e in events if isinstance(e, ResponseCompleted)), None)
        content = "".join(e.token for e in token_events)
        output_items = []
        if content:
            tool_call, cleaned_text = parse_xml_tool_call(content)
            if tool_call:
                output_items.append(
                    {
                        "type": "function_call",
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "name": tool_call["tool"],
                        "arguments": json.dumps(tool_call["arguments"]),
                    }
                )
                content = cleaned_text
        if content.strip():
            output_items.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                }
            )
        return {
            "id": start_event.response_id if start_event else f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created": int(start_event.timestamp if start_event else time.time()),
            "model": start_event.model_name if start_event else "unknown",
            "status": "completed",
            "output": output_items,
            "usage": {
                "input_tokens": start_event.input_tokens if start_event else 0,
                "output_tokens": end_event.output_tokens if end_event else len(token_events),
                "total_tokens": (start_event.input_tokens if start_event else 0)
                + (end_event.output_tokens if end_event else len(token_events)),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
