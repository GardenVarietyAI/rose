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

from rose_trainer.client import ServiceClient
from rose_trainer.fine_tuning.callbacks import CancellationCallback, EventCallback, HardwareMonitorCallback
from rose_trainer.fine_tuning.metrics import compute_perplexity
from rose_trainer.models import get_optimal_device, get_tokenizer, load_hf_model
from rose_trainer.types.fine_tuning import Hyperparameters

logger = logging.getLogger(__name__)


def train(
    job_id: str,
    model_name: str,
    training_file_path: Path,
    hyperparameters: Hyperparameters,
    client: ServiceClient,
    check_cancel_callback: Optional[Callable[[], str]] = None,
    event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a fine-tuning job and return a result dict."""

    # Fetch model info from API
    model_info = client.get_model(model_name)
    if not model_info:
        logger.error(
            f"Model '{model_name}' not found in models database. "
            "Please ensure the model is registered before fine-tuning."
        )
        raise ValueError(f"Model '{model_name}' not found")

    # Use the hyperparameters directly - no need to resolve
    torch.manual_seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hyperparameters.seed)

    # Use the actual HuggingFace model name from the database
    hf_model_name = model_info.get("model_name", model_name)
    config = config or {}
    model = load_hf_model(model_id=hf_model_name, config=config)
    tokenizer = get_tokenizer(hf_model_name, data_dir=config.get("data_dir", "./data"))

    # Extract config from model info
    model_config = {"lora_target_modules": model_info.get("lora_target_modules", [])}
    is_peft_model = False

    if hyperparameters.use_lora:
        lora_cfg = hyperparameters.lora_config or {}
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
    hardware_monitor = HardwareMonitorCallback(event_callback)
    callbacks: List[TrainerCallback] = [EventCallback(event_callback), hardware_monitor]
    if check_cancel_callback:
        checkpoint_dir_config = config.get("checkpoint_dir", "data/checkpoints")
        callbacks.append(CancellationCallback(check_cancel_callback, job_id, checkpoint_dir=checkpoint_dir_config))
    if hyperparameters.validation_split > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.early_stopping_patience))

    # Load dataset and tokenize
    try:
        raw_dataset: Dataset = load_dataset("json", data_files=str(training_file_path), split="train")
    except Exception as e:
        raise ValueError(
            "Failed to load training data: Invalid JSONL format. Each line must be a valid JSON object."
        ) from e

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
        result: BatchEncoding = tokenizer(str(text), truncation=True, max_length=hyperparameters.max_length)
        return result

    checkpoint_base_dir = config.get("checkpoint_dir", "data/checkpoints")
    checkpoint_dir = Path(checkpoint_base_dir) / job_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenized_dataset = raw_dataset.map(tokenize_example, remove_columns=raw_dataset.column_names)

    n_samples = len(tokenized_dataset)
    per_device = hyperparameters.batch_size
    steps_per_epoch = max(n_samples // per_device, 1)
    total_steps = steps_per_epoch // hyperparameters.gradient_accumulation_steps * hyperparameters.n_epochs
    device = get_optimal_device()

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=hyperparameters.n_epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=hyperparameters.eval_batch_size or 1,
        gradient_accumulation_steps=hyperparameters.gradient_accumulation_steps,
        learning_rate=hyperparameters.learning_rate,
        warmup_steps=int(total_steps * hyperparameters.warmup_ratio),
        lr_scheduler_type=hyperparameters.scheduler_type,
        eval_strategy="epoch" if hyperparameters.validation_split else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=bool(hyperparameters.validation_split),
        metric_for_best_model="eval_loss",  # Will be used to compute perplexity
        greater_is_better=False,
        optim="adamw_torch",
        weight_decay=hyperparameters.weight_decay,
        logging_dir=str(checkpoint_dir / "logs"),
        logging_steps=10,
        log_level="info",
        disable_tqdm=True,
        fp16=hyperparameters.fp16 if hyperparameters.fp16 is not None else (device == "cuda"),
        seed=hyperparameters.seed,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=(device == "cuda"),
        include_num_input_tokens_seen=True,
    )

    # tokenization
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # Build train / validation split
    if hyperparameters.validation_split > 0:
        split = tokenized_dataset.train_test_split(hyperparameters.validation_split, seed=42)
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
    if hyperparameters.suffix:
        model_id = f"{model_id}-{hyperparameters.suffix}"

    out_dir = Path(config.get("data_dir", "./data")) / "models" / model_id
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

    # Calculate perplexity from validation loss when available
    final_perplexity = None

    if hyperparameters.validation_split > 0 and eval_ds is not None:
        # The trainer.train() result doesn't include eval_loss in its metrics,
        # so we need to run evaluation to get the final validation loss
        eval_metrics = trainer.evaluate()

        if "eval_loss" in eval_metrics:
            final_perplexity = compute_perplexity(eval_metrics["eval_loss"])
            logger.info(f"Final validation perplexity: {final_perplexity:.4f} (loss: {eval_metrics['eval_loss']:.4f})")
        else:
            logger.error(
                f"Expected eval_loss in evaluation metrics but found: {list(eval_metrics.keys())}. "
                "This indicates a potential issue with the evaluation dataset or configuration."
            )

    # Get peak memory usage from hardware monitor
    peak_memory = hardware_monitor.get_peak_memory_gb()

    result_dict = {
        "success": True,
        "final_loss": result.metrics.get("train_loss"),
        "final_perplexity": final_perplexity,
        "steps": trainer.state.global_step,
        "tokens_processed": trainer.state.num_input_tokens_seen,
        "model_path": str(out_dir),
        "model_name": out_dir.name,
    }

    # Add peak memory metrics if available
    result_dict.update(peak_memory)

    return result_dict
