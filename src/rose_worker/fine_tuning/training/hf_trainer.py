import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

from rose_core.config.service import DATA_DIR, FINE_TUNING_EVAL_BATCH_SIZE, FINE_TUNING_MODELS
from rose_core.models import get_tokenizer, load_hf_model
from rose_core.models.loading import get_optimal_device

from .callbacks import CancellationCallback, EventCallback, HardwareMonitorCallback
from .hyperparams import HyperParams

logger = logging.getLogger(__name__)


def train(
    job_id: str,
    model_name: str,
    training_file_path: Path,
    hyperparameters: Dict[str, Any],
    check_cancel_callback: Optional[Callable[[], str]] = None,
    event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run a fine-tuning job and return a result dict."""

    if model_name not in FINE_TUNING_MODELS:
        raise ValueError(f"Model {model_name} not supported for fine-tuning")

    hp = HyperParams.resolve(hyperparameters)
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)

    hf_model_name = FINE_TUNING_MODELS[model_name]
    model = load_hf_model(model_id=hf_model_name)
    tokenizer = get_tokenizer(hf_model_name)
    model_config = FINE_TUNING_MODELS.get(model_name, {})

    if hp.use_lora and hp.lora_config:
        target_modules = hp.lora_config.get("target_modules")
        if not target_modules:
            target_modules = model_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

        peft_model = get_peft_model(  # type: ignore[arg-type]
            model,
            LoraConfig(
                r=hp.lora_config.get("r", 16),
                lora_alpha=hp.lora_config.get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=hp.lora_config.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )

        if not isinstance(peft_model, PeftModel):
            raise TypeError(f"Expected {type(peft_model).__name__} to be an instance of 'PeftModel'.")

        peft_model.print_trainable_parameters()

        model = peft_model

    # Build training components
    callbacks: List[TrainerCallback] = [EventCallback(event_callback), HardwareMonitorCallback(event_callback)]
    if check_cancel_callback:
        callbacks.append(CancellationCallback(check_cancel_callback, job_id))
    if hp.validation_split > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=hp.early_stopping_patience))

    # Load dataset and tokenize
    raw_dataset: Dataset = load_dataset("json", data_files=str(training_file_path), split="train")

    def tokenize_example(example: Dict[str, Any]) -> "BatchEncoding":
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(str(text), truncation=True, max_length=hp.max_length)

    checkpoint_dir = Path(DATA_DIR) / "checkpoints" / job_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenized_dataset = raw_dataset.map(tokenize_example, remove_columns=raw_dataset.column_names)

    n_samples = len(tokenized_dataset)
    per_device = hp.batch_size
    steps_per_epoch = max(n_samples // per_device, 1)
    total_steps = steps_per_epoch // hp.gradient_accumulation_steps * hp.n_epochs
    device = get_optimal_device()

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=hp.n_epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=FINE_TUNING_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        learning_rate=hp.learning_rate,
        warmup_steps=int(total_steps * hp.warmup_ratio),
        lr_scheduler_type=hp.scheduler_type,
        eval_strategy="epoch" if hp.validation_split else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=bool(hp.validation_split),
        metric_for_best_model="loss",
        optim="adamw_torch",
        logging_dir=str(checkpoint_dir / "logs"),
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

    latest = max(
        (p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()),
        default=None,
        key=lambda p: int(p.name.split("-")[1]),
    )
    result = trainer.train(resume_from_checkpoint=str(latest) if latest else None)

    ts = int(time.time())
    model_id = f"{model_name}-ft-{ts}"
    if hp.suffix:
        model_id = f"{model_id}-{hp.suffix}"

    out_dir = Path(DATA_DIR) / "models" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out_dir))

    # Check if this is a PEFT model
    if hasattr(trainer.model, "peft_config"):
        if torch.backends.mps.is_available():
            logger.warning("Skipping merge_and_unload on Apple Silicon")
        else:
            logger.info("Merging LoRA adapters into base model...")
            merged_model = trainer.model.merge_and_unload()  # type: ignore[union-attr,operator]
            merged_model.save_pretrained(str(out_dir))
            logger.info("Successfully merged and saved model")

    return {
        "success": True,
        "final_loss": result.metrics.get("train_loss"),
        "steps": trainer.state.global_step,
        "tokens_processed": trainer.state.num_input_tokens_seen,
        "model_path": str(out_dir),
        "model_name": out_dir.name,
    }
