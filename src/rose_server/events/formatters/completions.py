"""Completions API formatter for LLM events."""
import time
import uuid
from typing import Any, Dict, List, Optional
from .. import LLMEvent, ResponseCompleted, ResponseStarted, TokenGenerated

class CompletionsFormatter:
    """Formats events for OpenAI Completions API compatibility.

    This formatter converts LLM events to the legacy completions format
    which uses simpler text chunks rather than chat messages.
    """

    def __init__(self):
        self.completion_id: Optional[str] = None
        self.created_at: Optional[int] = None
        self.model_name: Optional[str] = None
        self.collected_text: str = ""

    def format_event(self, event: LLMEvent) -> Optional[Dict[str, Any]]:
        """Convert an LLM event to Completions API streaming format."""

        if isinstance(event, ResponseStarted):
            self.completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
            self.created_at = int(event.timestamp)
            self.model_name = event.model_name
            self.collected_text = ""
            return None
        elif isinstance(event, TokenGenerated):
            self.collected_text += event.token
            return {
                "id": self.completion_id or f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": self.created_at or int(time.time()),
                "choices": [{
                    "text": event.token,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None
                }],
                "model": self.model_name or event.model_name
            }
        elif isinstance(event, ResponseCompleted):
            return {
                "id": self.completion_id or f"cmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": self.created_at or int(time.time()),
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "model": self.model_name or event.model_name
            }
        return None

    def format_complete_response(self, events: List[LLMEvent]) -> Dict[str, Any]:
        """Format a complete (non-streaming) response from all events."""
        start_event = next((e for e in events if isinstance(e, ResponseStarted)), None)
        token_events = [e for e in events if isinstance(e, TokenGenerated)]
        end_event = next((e for e in events if isinstance(e, ResponseCompleted)), None)
        text = "".join(e.token for e in token_events)
        prompt_tokens = start_event.input_tokens if start_event else 0
        completion_tokens = end_event.output_tokens if end_event else len(token_events)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(start_event.timestamp if start_event else time.time()),
            "model": start_event.model_name if start_event else "unknown",
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }