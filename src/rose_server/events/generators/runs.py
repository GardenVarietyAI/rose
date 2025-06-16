"""Runs event generator for assistant execution."""
from typing import List, Optional
from ...schemas.chat import ChatMessage
from ...tools.formatter import format_tools_for_prompt
from .base import BaseEventGenerator

class RunsGenerator(BaseEventGenerator):
    """Generate events for assistant runs.

    This replaces the fake streaming in runs/streaming.py with real
    event-based streaming using the same infrastructure as chat completions.
    """

    async def prepare_prompt(
        self, 
        messages: List[ChatMessage], 
        enable_tools: bool = False,
        tools: Optional[List] = None
    ) -> str:
        """Prepare prompt for assistant runs.
        Assistant runs use the same format as chat completions,
        including tool support when needed.
        """
        msgs = list(messages)
        if enable_tools and tools:
            tool_prompt = format_tools_for_prompt(tools)
            if tool_prompt:
                sys_msg = ChatMessage(role="system", content=tool_prompt)
                insert_idx = 0
                for i, m in enumerate(msgs):
                    if m.role == "system":
                        insert_idx = i + 1
                msgs.insert(insert_idx, sys_msg)
        return self.llm.format_messages(msgs)