import logging
from typing import Any, Callable, Dict, Optional

import psutil
import torch
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from rose_trainer.fine_tuning.training.callbacks.base import _BaseCallback

logger = logging.getLogger(__name__)


class HardwareMonitorCallback(_BaseCallback):
    """Simple hardware monitoring callback."""

    def __init__(self, event_cb: Optional[Callable[[str, str, Dict[str, Any]], None]] = None) -> None:
        super().__init__(event_cb)
        self._process: Optional[psutil.Process] = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._process = psutil.Process()
        self._process.cpu_percent(None)

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
        if torch.cuda.is_available():
            pass  # CUDA monitoring not implemented

        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / 1024**3
                metrics["mps_memory_gb"] = round(allocated, 2)
            except Exception:
                pass

        if self._process:
            metrics["cpu_percent"] = self._process.cpu_percent()
            metrics["ram_gb"] = round(self._process.memory_info().rss / 1024**3, 2)

        self._send("debug", "Hardware metrics", metrics)
