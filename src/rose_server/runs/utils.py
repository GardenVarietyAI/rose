"""Utility functions for runs processing."""
from typing import Optional


def find_latest_user_message(messages) -> Optional[str]:
    """Find the latest user message from thread messages."""
    for msg in reversed(messages):
        if msg.role == "user":
            for content_item in msg.content:
                if content_item.type == "text" and content_item.text.value:
                    return content_item.text.value
    return None