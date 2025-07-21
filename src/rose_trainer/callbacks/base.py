import logging
import time
from typing import Any, Callable, Dict, Optional

from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class _BaseCallback(TrainerCallback):
    """Shared utilities for custom callbacks."""

    def __init__(self, event_cb: Optional[Callable[[str, str, Dict[str, Any]], None]] = None) -> None:
        self.event_cb = event_cb

    def _send(self, level: str, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self.event_cb:
            self.event_cb(level, msg, data or {})

    def _eta(self, done: int, total: int) -> str:
        if not self._t0 or done == 0:
            return "--:--"
        seconds = (time.time() - self._t0) / done * (total - done)
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
