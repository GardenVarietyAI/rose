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

from rose_core.config.settings import settings
from rose_core.models import get_tokenizer, load_hf_model
from rose_core.models.loading import get_optimal_device
from rose_trainer.client import ServiceClient
from rose_trainer.fine_tuning.training.callbacks import CancellationCallback, EventCallback, HardwareMonitorCallback
from rose_trainer.fine_tuning.training.hyperparams import HyperParams

logger = logging.getLogger(__name__)


def train(
    job_id: str,
    model_name: str,
    training_file_path: Path,
    hyperparameters: Dict[str, Any],
    client: ServiceClient,
    check_cancel_callback: Optional[Callable[[], str]] = None,
    event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run a fine-tuning job and return a result dict."""

    # Fetch model info from API
    model_info = client.get_model(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found")

    hp = HyperParams.resolve(hyperparameters)
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)

    # Use the actual HuggingFace model name from the database
    hf_model_name = model_info.get("model_name", model_name)
    model = load_hf_model(model_id=hf_model_name)
    tokenizer = get_tokenizer(hf_model_name)

    # Extract config from model info
    model_config = {"lora_target_modules": model_info.get("lora_target_modules", [])}
    is_peft_model = False

    if hp.use_lora:
        lora_cfg = hp.lora_config or {}
        target_modules = lora_cfg.get("target_modules")
        if not target_modules:
            target_modules = model_config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

        peft_model = get_peft_model(
            model,
            LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )

        if not isinstance(peft_model, PeftModel):
            raise TypeError(f"Expected {type(peft_model).__name__} to be an instance of 'PeftModel'.")

        peft_model.print_trainable_parameters()

        model = peft_model
        is_peft_model = True

    # Build training components
    callbacks: List[TrainerCallback] = [EventCallback(event_callback), HardwareMonitorCallback(event_callback)]
    if check_cancel_callback:
        callbacks.append(CancellationCallback(check_cancel_callback, job_id))
    if hp.validation_split > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=hp.early_stopping_patience))

    # Load dataset and tokenize
    raw_dataset: Dataset = load_dataset("json", data_files=str(training_file_path), split="train")

    def tokenize_example(example: Dict[str, Any]) -> "BatchEncoding":
        # Check if tokenizer has a chat template
        if tokenizer.chat_template is None:
            # Fallback for models without chat templates (like phi-1.5)
            messages = example["messages"]
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    text_parts.append(f"System: {content}")
                elif role == "user":
                    text_parts.append(f"User: {content}")
                elif role == "assistant":
                    text_parts.append(f"Assistant: {content}")
            text = "\n".join(text_parts)
        else:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        result: BatchEncoding = tokenizer(str(text), truncation=True, max_length=hp.max_length)
        return result

    checkpoint_dir = Path(settings.data_dir) / "checkpoints" / job_id
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
        per_device_eval_batch_size=settings.fine_tuning_eval_batch_size,
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

    out_dir = Path(settings.data_dir) / "models" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is a PEFT model
    if is_peft_model:
        if torch.backends.mps.is_available():
            logger.info("Saving LoRA adapter separately on Apple Silicon...")
            # Save only the PEFT adapter, not the full model
            trainer.model.save_pretrained(str(out_dir))
            # Also save the tokenizer so it can be loaded with the adapter
            tokenizer.save_pretrained(str(out_dir))
            logger.info("Successfully saved LoRA adapter")
        else:
            logger.info("Merging LoRA adapters into base model...")
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(out_dir))
            # Save tokenizer with the merged model
            tokenizer.save_pretrained(str(out_dir))
            logger.info("Successfully merged and saved model")
    else:
        # Not a PEFT model, save normally
        trainer.save_model(str(out_dir))

    return {
        "success": True,
        "final_loss": result.metrics.get("train_loss"),
        "steps": trainer.state.global_step,
        "tokens_processed": trainer.state.num_input_tokens_seen,
        "model_path": str(out_dir),
        "model_name": out_dir.name,
    }
