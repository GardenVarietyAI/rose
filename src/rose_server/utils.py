import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def extract_user_content(content: Union[str, List[Dict[str, Any]]]) -> Optional[str]:
    """Extract text content from various content formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        for content_item in content:
            if content_item.get("type") == "input_text":
                return content_item.get("text", "")
    return None
