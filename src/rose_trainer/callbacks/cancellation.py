import json
import logging
from pathlib import Path
from typing import Any, Callable

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class CancellationCallback(TrainerCallback):
    """Stops training early when an external controller requests it."""

    def __init__(
        self,
        status_fn: Callable[[], str],
        job_id: str,
        checkpoint_dir: str = "data/checkpoints",
    ) -> None:
        self._status_fn = status_fn
        self._job_id = job_id
        self._checkpoint_dir = checkpoint_dir
        self.cancelled = False
        self.paused = False

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        match self._status_fn():
            case "cancelling":
                self.cancelled, control.should_training_stop = True, True
            case "pausing":
                self.paused, control.should_save, control.should_training_stop = True, True, True
        return control

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        if self.paused:
            meta = {
                "is_paused": True,
                "global_step": state.global_step,
                "epoch": state.epoch,
            }
            pause_meta_path = Path(self._checkpoint_dir) / self._job_id / "pause_meta.json"
            pause_meta_path.parent.mkdir(parents=True, exist_ok=True)
            pause_meta_path.write_text(json.dumps(meta))
        return control
