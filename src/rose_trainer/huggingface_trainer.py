import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from peft.peft_model import PeftModel
from torch.optim import Optimizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from rose_trainer.types import Hyperparameters

logger = logging.getLogger(__name__)


class HuggingfaceTrainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizerBase,
        is_peft: bool,
        hyperparams: Hyperparameters,
        device: str,
        checkpoint_dir: Path,
        callbacks: List[TrainerCallback],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.is_peft = is_peft
        self.hyperparams = hyperparams
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.callbacks = callbacks

        # Training components (initialized in prepare_training)
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[Any] = None

        # Training state
        self.global_step: int = 0
        self.total_tokens: int = 0
        self.total_steps: int = 0
        self.start_time: Optional[float] = None
        self.final_loss = 0.0
        self.success: bool = False

    def train(self, dataset: Dataset) -> None:
        """Train the model on the dataset."""

        # Start timer
        self.start_time = time.time()

        # Split dataset
        if self.hyperparams.validation_split > 0:
            split = dataset.train_test_split(self.hyperparams.validation_split, seed=self.hyperparams.seed)
            train_ds = split["train"]
            eval_ds = split["test"]
        else:
            train_ds = dataset
            eval_ds = None

        n_samples = len(dataset)
        per_device = self.hyperparams.batch_size
        steps_per_epoch = max(n_samples // per_device, 1)
        total_steps = steps_per_epoch // self.hyperparams.gradient_accumulation_steps * self.hyperparams.n_epochs

        # Training args
        args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=self.hyperparams.n_epochs,
            per_device_train_batch_size=per_device,
            per_device_eval_batch_size=self.hyperparams.eval_batch_size or 1,
            gradient_accumulation_steps=self.hyperparams.gradient_accumulation_steps,
            learning_rate=self.hyperparams.learning_rate,
            warmup_steps=int(total_steps * self.hyperparams.warmup_ratio),
            lr_scheduler_type=self.hyperparams.scheduler_type,
            eval_strategy="epoch" if self.hyperparams.validation_split else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=bool(self.hyperparams.validation_split),
            metric_for_best_model="eval_loss",  # Will be used to compute perplexity
            greater_is_better=False,
            optim="adamw_torch",
            weight_decay=self.hyperparams.weight_decay,
            logging_dir=str(self.checkpoint_dir / "logs"),
            logging_steps=10,
            log_level="info",
            disable_tqdm=True,
            fp16=self.hyperparams.fp16 if self.hyperparams.fp16 is not None else (self.device == "cuda"),
            seed=self.hyperparams.seed,
            auto_find_batch_size=True,
            remove_unused_columns=False,
            report_to=[],
            dataloader_pin_memory=(self.device == "cuda"),
            include_num_input_tokens_seen=True,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
            callbacks=self.callbacks,
        )

        # Train
        result = trainer.train()

        # Validate training actually succeeded
        expected_steps = len(trainer.get_train_dataloader()) * self.hyperparams.n_epochs
        actual_steps = trainer.state.global_step

        if actual_steps == 0:
            raise RuntimeError("Training failed: no steps completed")

        if actual_steps < expected_steps * 0.1:  # Less than 10% of expected steps
            raise RuntimeError(f"Training incomplete: {actual_steps}/{expected_steps} steps")

        final_loss = result.metrics.get("train_loss")
        if final_loss is None or final_loss <= 0:
            raise RuntimeError("Training failed: invalid final loss")

        # Training succeeded
        self.success = True
        self.final_loss = final_loss
        self.global_step = trainer.state.global_step
        self.total_tokens = trainer.state.num_input_tokens_seen or 0

    def save(self, output_dir: Path, base_model_id: str, model_name: str) -> Dict[str, Any]:
        """Save model and return training metadata."""
        # Calculate final metrics
        training_time = time.time() - self.start_time if self.start_time else 0

        # Save metadata
        metadata = {
            "base_model": base_model_id,
            "trainer": "huggingface",
            "hyperparameters": self.hyperparams.model_dump(),
            "training_time": training_time,
            "global_steps": self.global_step,
            "final_loss": self.final_loss,
            "steps": self.global_step,
            "tokens_processed": self.total_tokens,
            "model_path": str(output_dir),
            "model_name": model_name,
            "success": True,
        }

        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
