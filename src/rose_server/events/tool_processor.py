"""Tool detection and event generation for streaming responses."""

import json
import uuid
from typing import List, Optional, Tuple, Union

from rose_server.events.event_types import (
    TokenGenerated,
    ToolCallCompleted,
    ToolCallStarted,
)
from rose_server.tools import StreamingXMLDetector


class ToolProcessor:
    """Handles tool detection in token streams."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.detector = StreamingXMLDetector()

    def process_token(
        self, event: TokenGenerated
    ) -> Tuple[List[Union[ToolCallStarted, ToolCallCompleted]], Optional[TokenGenerated]]:
        """Process a token event for tool calls.

        Returns:
            - List of tool events to emit
            - Modified token event (or None if token was consumed by tool)
        """
        tool_events = []
        plain_text, tool_call = self.detector.process_token(event.token)

        if tool_call:
            call_id = f"call_{uuid.uuid4().hex[:16]}"

            tool_events.extend(
                [
                    ToolCallStarted(
                        model_name=self.model_name,
                        function_name=tool_call["tool"],
                        call_id=call_id,
                        arguments_so_far="",
                    ),
                    ToolCallCompleted(
                        model_name=self.model_name,
                        function_name=tool_call["tool"],
                        call_id=call_id,
                        arguments=json.dumps(tool_call["arguments"]),
                    ),
                ]
            )

        # Return modified event if there's plain text
        if plain_text:
            event.token = plain_text
            return tool_events, event
        else:
            return tool_events, None
