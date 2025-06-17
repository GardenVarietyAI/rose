import logging

import psutil
import torch

from .base import _BaseCallback

logger = logging.getLogger(__name__)


class HardwareMonitorCallback(_BaseCallback):
    """Simple hardware monitoring callback."""

    def on_log(self, args, state, control, logs=None, **_):
        if not logs or state.global_step % 10 != 0:
            return

        metrics = {}
        if torch.cuda.is_available():
            try:
                logger.warning("Not implemented")
            except Exception:
                pass

        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / 1024**3
                metrics["mps_memory_gb"] = round(allocated, 2)
            except Exception:
                pass

        process = psutil.Process()
        metrics["cpu_percent"] = process.cpu_percent()
        metrics["ram_gb"] = round(process.memory_info().rss / 1024**3, 2)

        self._send("debug", "Hardware metrics", metrics)
