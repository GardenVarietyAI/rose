import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from peft.peft_model import PeftModel
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

from rose_trainer.perplexity import compute_perplexity
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
        self.optimizer: str = "adamw_torch"
        self.default_collator: DataCollatorForLanguageModeling = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Training state
        self.global_step: int = 0
        self.total_tokens: int = 0
        self.start_time: Optional[float] = None
        self.final_loss = 0.0
        self.final_perplexity = None
        self.success: bool = False

    def _validate_batch(self, batch: Dict[str, Any], tokenizer_pad_id: int) -> None:
        """Validate batch to catch tokenizer issues early."""
        if "labels" in batch:
            # Check if padding is dominating the loss
            labels = batch["labels"]
            pad_count = (labels == -100).sum().item()
            total_count = labels.numel()
            pad_ratio = pad_count / total_count if total_count > 0 else 0

            if pad_ratio > 0.9:
                logger.warning(f"High padding ratio in batch: {pad_ratio:.2%} tokens are padding")

        # Check attention mask alignment
        if "attention_mask" in batch and "input_ids" in batch:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Verify shapes match
            if input_ids.shape != attention_mask.shape:
                logger.error(f"Shape mismatch: input_ids {input_ids.shape} vs attention_mask {attention_mask.shape}")

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

        args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=self.hyperparams.n_epochs,
            per_device_train_batch_size=per_device,
            per_device_eval_batch_size=1 if self.device == "cpu" else (self.hyperparams.eval_batch_size or 1),
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
            optim=self.optimizer,
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

        def validating_collator(*args, **kwargs):
            batch = self.default_collator(*args, **kwargs)
            self._validate_batch(batch, self.tokenizer.pad_token_id)
            return batch

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            data_collator=validating_collator,
            callbacks=self.callbacks,
        )

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
        # Calculate perplexity from eval_loss if available
        eval_loss = result.metrics.get("eval_loss")
        self.final_perplexity = compute_perplexity(eval_loss) if eval_loss else None
        self.global_step = trainer.state.global_step
        self.total_tokens = trainer.state.num_input_tokens_seen or 0
        self.epochs_completed = trainer.state.epoch

    def save(self, output_dir: Path, base_model_id: str, model_name: str) -> Dict[str, Any]:
        """Save model and return training metadata."""
        # Calculate final metrics
        finish_time = time.time()
        training_time = finish_time - self.start_time if self.start_time else 0

        # Save metadata
        metadata = {
            "base_model": base_model_id,
            "trainer": "huggingface",
            "hyperparameters": self.hyperparams.model_dump(),
            "training_time": training_time,
            "global_steps": self.global_step,
            "final_loss": self.final_loss,
            "final_perplexity": self.final_perplexity,
            "start_time": self.start_time,
            "finish_time": finish_time,
            "steps": self.global_step,
            "tokens_processed": self.total_tokens,
            "epochs_completed": self.epochs_completed,
            "model_path": str(output_dir),
            "model_name": model_name,
            "success": True,
        }

        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
