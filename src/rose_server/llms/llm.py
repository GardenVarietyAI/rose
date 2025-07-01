"""Lightweight LLM wrapper for WebSocket inference."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLM:
    """
    Lightweight wrapper for WebSocket-based inference.
    All formatting and tokenization happens in the inference worker.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.model_name: str = config.get("model_name", "unknown")
        self.model_path: Optional[str] = config.get("model_path")

    @property
    def model(self) -> None:
        """No local model - inference happens via WebSocket."""
        return None

    @property
    def tokenizer(self) -> None:
        """No local tokenizer - tokenization happens in inference worker."""
        return None

    def cleanup(self) -> None:
        """No cleanup needed since we don't load models."""
        pass
