"""Responses event generator for the /v1/responses API."""
from typing import List, Optional
from ...schemas.chat import ChatMessage
from .base import BaseEventGenerator

class ResponsesGenerator(BaseEventGenerator):
    """Generate events for the Responses API.

    The Responses API is similar to chat completions but with:
    - Different response format (response.* events)
    - Support for instructions in addition to messages
    - Store functionality for persisting responses
    """

    async def prepare_prompt(
        self, 
        messages: List[ChatMessage], 
        enable_tools: bool = False,
        tools: Optional[List] = None
    ) -> str:
        """Prepare prompt for responses.
        The responses endpoint handles system prompts and tools similarly
        to chat completions, so we can reuse the same formatting.
        """
        if enable_tools and tools:
            return self.llm.format_messages_with_tools(messages, tools)
        else:
            return self.llm.format_messages(messages)