import logging
import time
from typing import Any, Dict, Optional

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from rose_trainer.fine_tuning.callbacks.base import _BaseCallback

logger = logging.getLogger(__name__)


class EventCallback(_BaseCallback):  # type: ignore[misc]
    """Streams high-level training progress to ``event_callback``."""

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._t0 = time.time()

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        epoch = state.epoch if state.epoch is not None else 0
        self._send(
            "info",
            f"Epoch {epoch + 1}/{args.num_train_epochs} started",
            {"epoch": epoch + 1},
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        total = args.max_steps
        if total <= 0:
            if hasattr(state, "num_train_epochs") and hasattr(args, "num_train_epochs"):
                if state.global_step > 0 and state.epoch is not None and state.epoch > 0:
                    steps_per_epoch = int(state.global_step / state.epoch)
                    total = int(steps_per_epoch * args.num_train_epochs)
                else:
                    total = 0
            else:
                total = 0
        pct = 100 * state.global_step / total if total > 0 else 0
        self._send(
            "info",
            f"Step {state.global_step}/{total} "
            f"({pct:.1f} %) - loss {logs['loss']:.4f} - ETA {self._eta(state.global_step, total)}",
            {
                "step": state.global_step,
                "loss": logs["loss"],
                "progress_pct": round(pct, 2),
            },
        )
