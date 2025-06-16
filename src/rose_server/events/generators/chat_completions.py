"""Chat completions event generator."""
from typing import List, Optional

from ...schemas.chat import ChatMessage
from ...tools.formatter import format_tools_for_prompt
from .base import BaseEventGenerator


class ChatCompletionsGenerator(BaseEventGenerator):
    """Generate events for chat completions with tool support."""

    async def prepare_prompt(
        self, 
        messages: List[ChatMessage], 
        enable_tools: bool = False,
        tools: Optional[List] = None
    ) -> str:
        """Prepare prompt with optional tool instructions."""
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