import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.training_args import TrainingArguments

from rose_core.config.service import DATA_DIR, FINE_TUNING_EVAL_BATCH_SIZE, FINE_TUNING_MODELS, LLM_MODELS
from rose_core.models import cleanup_model_memory, get_tokenizer, load_hf_model
from rose_core.models.loading import get_optimal_device

from .callbacks import CancellationCallback, EventCallback, HardwareMonitorCallback
from .hyperparams import HyperParams

logger = logging.getLogger(__name__)


def resolve_hyperparams(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve auto values to concrete numbers."""
    return {
        "batch_size": int(raw.get("batch_size", 1)),
        "max_length": int(raw.get("max_length", 512)),
        "n_epochs": int(raw.get("n_epochs", 3)),
        "learning_rate": float(raw.get("learning_rate", 5e-5)),
        "gradient_accumulation_steps": int(raw.get("gradient_accumulation_steps", 1)),
        "validation_split": float(raw.get("validation_split", 0.1)),
        "early_stopping_patience": int(raw.get("early_stopping_patience", 3)),
        "warmup_ratio": float(raw.get("warmup_ratio", 0.1)),
        "scheduler_type": raw.get("scheduler_type", "cosine"),
        "min_lr_ratio": float(raw.get("min_lr_ratio", 0.1)),
        "use_lora": bool(raw.get("use_lora", True)),
        "lora_config": raw.get("lora_config"),
        "seed": int(raw.get("seed", 42)),
        "suffix": raw.get("suffix", "custom"),
    }


class HFTrainer:
    """Wraps HF Trainer with checkpoint management and resource monitoring."""

    def __init__(self) -> None:
        self.fine_tuning_models = FINE_TUNING_MODELS.copy()

    def train(
        self,
        job_id: str,
        model_name: str,
        training_file_path: Path,
        hyperparameters: Dict[str, Any],
        check_cancel_callback: Optional[Callable[[], str]] = None,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run a fine-tuning job and return a result dict."""

        if model_name not in self.fine_tuning_models:
            raise ValueError(f"Model {model_name} not supported for fine-tuning")

        hp = HyperParams.resolve(hyperparameters)
        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(hp.seed)

        hf_model_name = self.fine_tuning_models[model_name]
        model = load_hf_model(model_id=hf_model_name)
        tokenizer = get_tokenizer(hf_model_name)

        if hp.use_lora:
            model = self._apply_lora(model, model_name, hp)

        # Build training components
        callbacks: List[TrainerCallback] = [EventCallback(event_callback), HardwareMonitorCallback(event_callback)]
        if check_cancel_callback:
            callbacks.append(CancellationCallback(check_cancel_callback, job_id))
        if hp.validation_split > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hp.early_stopping_patience))

        # Load dataset and tokenize
        raw_dataset: Dataset = load_dataset("json", data_files=str(training_file_path), split="train")  # type: ignore[arg-type]

        def tokenize_example(example: Dict[str, Any]) -> "BatchEncoding":
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            return tokenizer(str(text), truncation=True, model_max_length=hp.max_length)  # type: ignore[arg-type]

        tokenized_dataset = raw_dataset.map(tokenize_example, remove_columns=raw_dataset.column_names)  # type: ignore[arg-type]

        args = _make_training_args(job_id, hp, len(tokenized_dataset))

        # tokenization
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

        # Build train / validation split
        if hp.validation_split > 0:
            split = tokenized_dataset.train_test_split(hp.validation_split, seed=42)
            train_ds = split["train"]
            eval_ds = split["test"]
        else:
            train_ds = tokenized_dataset
            eval_ds = None

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        try:
            self._log_training_start(event_callback, model_name, tokenized_dataset, args)
            result = trainer.train(resume_from_checkpoint=_latest_checkpoint(job_id))
            out_dir = self._save_model(trainer, model_name, hp.suffix)
            metrics = result.metrics
            tokens = trainer.state.num_input_tokens_seen

            return {
                "success": True,
                "final_loss": metrics.get("train_loss", 0),
                "steps": trainer.state.global_step,
                "tokens_processed": tokens,
                "model_path": str(out_dir),
                "model_name": out_dir.name,
            }
        except Exception:
            logger.exception("Training failed")
            raise
        finally:
            cleanup_model_memory(model)

    def _log_training_start(
        self,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]],
        model_name: str,
        dataset: Dataset,
        args: TrainingArguments,
    ) -> None:
        if event_callback:
            event_callback(
                "info",
                "Training started",
                {
                    "model": model_name,
                    "num_examples": len(dataset),
                    "batch_size": args.per_device_train_batch_size,
                    "epochs": args.num_train_epochs,
                    "device": get_optimal_device(),
                },
            )

    def _apply_lora(self, model: "PreTrainedModel", model_name: str, hp: HyperParams) -> PeftModel:
        """Apply LoRA adaptation to the model."""
        lora_cfg = hp.lora_config or {}

        # Get target modules from config or model registry
        target_modules = lora_cfg.get("target_modules")
        if not target_modules:
            model_config = LLM_MODELS.get(model_name, {})
            target_modules = model_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, lora_config)  # type: ignore[arg-type]
        if not isinstance(peft_model, PeftModel):
            raise TypeError(
                "Expected 'peft_model' to be an instance of 'PeftModel', but got {type(peft_model).__name__}."
            )
        peft_model.print_trainable_parameters()

        return peft_model

    def _save_model(self, trainer: Trainer, base_name: str, suffix: str) -> Path:
        ts = int(time.time())

        if suffix:
            model_id = f"{base_name}-ft-{ts}-{suffix}"
        else:
            model_id = f"{base_name}-ft-{ts}"

        out = Path(DATA_DIR) / "models" / model_id
        out.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(out))

        # Check if this is a PEFT model
        if hasattr(trainer.model, "peft_config"):
            if torch.backends.mps.is_available():
                logger.warning(
                    "Skipping merge_and_unload on Apple Silicon due to potential hanging issues. "
                    "The model is saved with separate adapter weights."
                )
            else:
                try:
                    logger.info("Merging LoRA adapters into base model...")
                    merged_model = trainer.model.merge_and_unload()  # type: ignore[union-attr,operator]
                    merged_model.save_pretrained(str(out))
                    logger.info("Successfully merged and saved model")
                except Exception as e:
                    logger.error(f"Failed to merge_and_unload model: {e}")

        return out

    def cleanup(self) -> None:
        cleanup_model_memory()


def _make_training_args(job_id: str, hp: HyperParams, n_samples: int) -> TrainingArguments:
    out_dir = Path(DATA_DIR) / "checkpoints" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_device = hp.batch_size
    steps_per_epoch = max(n_samples // per_device, 1)
    actual_gas = hp.gradient_accumulation_steps
    total_steps = steps_per_epoch // actual_gas * hp.n_epochs
    warmup = int(total_steps * hp.warmup_ratio)
    device = get_optimal_device()

    return TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=hp.n_epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=FINE_TUNING_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=actual_gas,
        learning_rate=hp.learning_rate,
        warmup_steps=warmup,
        lr_scheduler_type=hp.scheduler_type,
        eval_strategy="epoch" if hp.validation_split else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=bool(hp.validation_split),
        metric_for_best_model="loss",
        optim="adamw_torch",
        logging_dir=str(out_dir / "logs"),
        logging_steps=10,
        log_level="info",
        disable_tqdm=True,
        fp16=(device == "cuda"),
        seed=hp.seed,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=(device == "cuda"),
        include_num_input_tokens_seen=True,
    )


def _latest_checkpoint(job_id: str) -> Optional[str]:
    base = Path(DATA_DIR) / "checkpoints" / job_id
    if not base.exists():
        return None
    ckpts = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        return str(ckpts[-1])
    return None
