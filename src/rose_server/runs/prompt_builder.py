import logging
from typing import List, Optional

from rose_server.entities.messages import Message
from rose_server.tools import format_tools_for_prompt

logger = logging.getLogger(__name__)


def find_latest_user_message(messages: List[Message]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        for item in msg.content:
            if item["type"] == "text":
                return str(item["text"]["value"])
    return None


def build_conversation_context(messages: List[Message], *, limit: int = 5) -> str:
    parts: List[str] = []
    for msg in messages[-limit:]:
        text_parts = []
        for item in msg.content:
            if item["type"] == "text":
                text_parts.append(item["text"]["value"])
        text = "".join(text_parts)
        if text:
            parts.append(f"{msg.role}: {text}")
    return "\n".join(parts)


async def build_prompt(
    *,
    instructions: Optional[str],
    messages: List[Message],
    latest_user_message: str,
    tools: Optional[List] = None,
    assistant_id: Optional[str] = None,
) -> str:
    prompt_parts: List[str] = []

    if instructions:
        prompt_parts.append(f"Instructions: {instructions}")

    context = build_conversation_context(messages)
    if context:
        prompt_parts.append(f"Recent conversation:\n{context}")

    if tools:
        tool_prompt = format_tools_for_prompt(tools, assistant_id=assistant_id)
        if tool_prompt:
            prompt_parts.append(tool_prompt)

    prompt_parts.append(f"\nUser: {latest_user_message}")
    return "\n\n".join(prompt_parts)
