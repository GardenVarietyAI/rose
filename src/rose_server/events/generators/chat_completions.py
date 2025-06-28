"""Chat completions event generator."""

from typing import List, Optional

from ...schemas.chat import ChatMessage
from .base import BaseEventGenerator


class ChatCompletionsGenerator(BaseEventGenerator):
    """Generate events for chat completions with tool support."""

    async def prepare_prompt(
        self, messages: List[ChatMessage], enable_tools: bool = False, tools: Optional[List] = None
    ) -> str:
        """Prepare prompt with optional tool instructions."""
        if enable_tools and tools:
            # Use the format_messages_with_tools method which handles tool formatting
            return self.llm.format_messages_with_tools(messages, tools)
        return self.llm.format_messages(messages)
