import logging
from typing import Any, Callable, Dict, Optional

import psutil
import torch
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from rose_trainer.callbacks.base import _BaseCallback

logger = logging.getLogger(__name__)


class HardwareMonitorCallback(_BaseCallback):  # type: ignore[misc]
    """Simple hardware monitoring callback."""

    def __init__(self, event_cb: Optional[Callable[[str, str, Dict[str, Any]], None]] = None) -> None:
        super().__init__(event_cb)
        self._process: Optional[psutil.Process] = None
        self._peak_memory_gb: float = 0.0
        self._device_type: Optional[str] = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(None)

        # Determine device type and reset memory stats
        if torch.cuda.is_available():
            self._device_type = "cuda"
            torch.cuda.reset_peak_memory_stats()
        elif torch.backends.mps.is_available():
            self._device_type = "mps"

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs or state.global_step % 10 != 0:
            return

        metrics = {}

        try:
            # Track memory based on device type
            if self._device_type == "cuda":
                current_memory_gb = torch.cuda.memory_allocated() / 1e9
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                metrics["cuda_memory_gb"] = round(current_memory_gb, 2)
                self._peak_memory_gb = max(self._peak_memory_gb, peak_memory_gb)

            elif self._device_type == "mps":
                current_memory_gb = torch.mps.current_allocated_memory() / 1e9
                metrics["mps_memory_gb"] = round(current_memory_gb, 2)
                self._peak_memory_gb = max(self._peak_memory_gb, current_memory_gb)
        except (RuntimeError, AttributeError):
            pass

        if self._process:
            metrics["cpu_percent"] = self._process.cpu_percent()
            metrics["ram_gb"] = round(self._process.memory_info().rss / 1024**3, 2)

        self._send("info", "Hardware metrics", metrics)

    def get_peak_memory_gb(self) -> Dict[str, float]:
        """Return peak memory usage for the device used during training."""
        if self._device_type and self._peak_memory_gb > 0:
            key = f"{self._device_type}_peak_memory_gb"
            return {key: round(self._peak_memory_gb, 2)}
        return {}
