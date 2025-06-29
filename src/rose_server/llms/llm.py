"""Lightweight LLM wrapper for WebSocket inference."""

import logging
from typing import Any, Dict, List, Optional

from rose_server.schemas.chat import ChatMessage
from rose_server.tools import format_tools_for_prompt

logger = logging.getLogger(__name__)


class LLM:
    """
    Lightweight wrapper that only handles prompt formatting.
    Actual model inference happens via WebSocket to the worker.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.model_name: str = config.get("model_name", "unknown")
        self.model_path: Optional[str] = config.get("model_path")

        # No model or tokenizer loading!
        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> None:
        return None

    @property
    def tokenizer(self) -> None:
        return None

    def format_messages_with_tools(self, messages: List[ChatMessage], tools: List[Any]) -> str:
        """Format messages with tool definitions included."""
        base_prompt = self.format_messages(messages)

        if tools:
            tool_prompt = format_tools_for_prompt(tools)
            if tool_prompt:
                return f"{tool_prompt}\n\n{base_prompt}"

        return base_prompt

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """
        Convert messages to a simple prompt format.
        Since we don't have the tokenizer, use a generic format.
        """
        prompt_parts: List[str] = []
        role_map = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "System",
            "function": "System",
            "developer": "System",
        }

        for msg in messages:
            text = self._extract_text(msg)
            if text is None:
                continue
            prompt_parts.append(f"{role_map.get(msg.role, 'User')}: {text}\n\n")

        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    @staticmethod
    def _extract_text(msg: ChatMessage) -> Optional[str]:
        """Extract plain text from message content."""
        if msg.content is None:
            return None
        if isinstance(msg.content, str):
            return msg.content
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in ("text", "input_text") and "text" in item:
                        return str(item["text"])
                    else:
                        logger.debug(f"Skipping non-text content part: type={item_type}")
        return None

    def cleanup(self) -> None:
        """No cleanup needed since we don't load models."""
        pass
