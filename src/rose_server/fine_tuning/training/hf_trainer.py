import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from transformers.training_args import TrainingArguments

from ...config import ServiceConfig
from ...hf.loading import load_model_and_tokenizer
from ...model_registry import FINE_TUNING_MODELS, get_model_config
from .callbacks import CancellationCallback, EventCallback, HardwareMonitorCallback
from .hyperparams import HyperParams

logger = logging.getLogger(__name__)


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

        os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

        raw_data = _load_jsonl(training_file_path)
        hp = HyperParams.resolve(hyperparameters)

        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(hp.seed)

        model, tokenizer = self._load_model_and_tok(model_name, hp)
        ds = _prepare_dataset(raw_data, tokenizer, hp.max_length)
        args = _make_training_args(job_id, hp, len(ds))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

        callbacks: List[TrainerCallback] = [
            EventCallback(event_callback),
            HardwareMonitorCallback(event_callback),
        ]

        if check_cancel_callback:
            callbacks.append(CancellationCallback(check_cancel_callback, job_id))

        if hp.validation_split > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hp.early_stopping_patience))

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            eval_dataset=_maybe_split(ds, hp.validation_split),
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        try:
            if event_callback:
                event_callback(
                    "info",
                    "Training started",
                    {
                        "model": model_name,
                        "num_examples": len(ds),
                        "batch_size": args.per_device_train_batch_size,
                        "epochs": args.num_train_epochs,
                        "device": "cuda"
                        if torch.cuda.is_available()
                        else "mps"
                        if torch.backends.mps.is_available()
                        else "cpu",
                    },
                )

            result = trainer.train(resume_from_checkpoint=_latest_checkpoint(job_id))
            out_dir = self._save_model(trainer, model_name, hp.suffix)
            metrics = result.metrics
            tokens = int(trainer.state.num_input_tokens_seen) if hasattr(trainer.state, "num_input_tokens_seen") else 0

            return {
                "success": True,
                "final_loss": metrics.get("train_loss", 0),
                "steps": trainer.state.global_step,
                "tokens_processed": tokens,
                "model_path": str(out_dir),
                "model_name": out_dir.name,
            }
        except Exception as exc:
            logger.exception("Training failed")
            return {"success": False, "error": str(exc)}
        finally:
            self.cleanup()

    def _load_model_and_tok(
        self, model_name: str, hp: HyperParams
    ) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizer]:
        """Load model with hyperparameters applied."""

        if model_name not in self.fine_tuning_models:
            raise ValueError(f"Model {model_name} not supported for fine-tuning")

        hf_model_name = self.fine_tuning_models[model_name]
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(
            model_id=hf_model_name,
            device=device,
            device_map=None,
            offload_dir=None,
        )

        if hp.use_lora:
            lora_cfg = hp.lora_config or {}
            # Get target modules from model registry if not specified
            target_modules = lora_cfg.get("target_modules")

            if not target_modules:
                model_config = get_model_config(model_name)
                target_modules = model_config.get("lora_target_modules", ["q_proj", "v_proj"])

            lora_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, lora_config)

            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        return model, tokenizer

    def _save_model(self, trainer: Trainer, base_name: str, suffix: str) -> Path:
        ts = int(time.time())

        if suffix and suffix != "None":
            model_id = f"{base_name}-ft-{ts}-{suffix}"
        else:
            model_id = f"{base_name}-ft-{ts}"

        out = Path(ServiceConfig.DATA_DIR) / "models" / model_id
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
                    merged_model = trainer.model.merge_and_unload()
                    merged_model.save_pretrained(str(out))
                    logger.info("Successfully merged and saved model")
                except Exception as e:
                    logger.error(f"Failed to merge_and_unload model: {e}")

        relative_path = out.relative_to(Path(ServiceConfig.DATA_DIR))
        self._update_registry(model_id, str(relative_path), base_name)
        return out

    def _update_registry(self, model_id: str, model_path: str, base_model: str):
        """Update the fine-tuned models registry."""
        import json

        registry_path = Path(ServiceConfig.DATA_DIR) / "fine_tuned_models.json"
        registry = {}
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")

        # Get the actual HuggingFace model name, not the simplified name
        hf_model_name = self.fine_tuning_models.get(base_model, base_model)

        registry[model_id] = {
            "path": model_path,
            "base_model": base_model,  # Keep simplified name for compatibility
            "hf_model_name": hf_model_name,  # Add actual HF name
            "created_at": time.time(),
        }

        try:
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Updated fine-tuned models registry with {model_id}")
        except Exception as e:
            logger.error(f"Failed to update registry: {e}")

    def cleanup(self):
        """Clean up all resources held by HFTrainer."""

        import gc

        gc.collect()


def _load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    """Read JSONL file, ignoring blank lines."""
    lines = [ln for ln in fp.read_text("utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _prepare_dataset(raw: Sequence[Dict[str, Any]], tokenizer: Any, max_len: int) -> Dataset:
    def to_text(item: Dict[str, Any]) -> str:
        if "messages" in item and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)

        if "prompt" in item and "completion" in item:
            return item["prompt"] + item["completion"]

        return item.get("text", "")

    texts = [to_text(ex) for ex in raw]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=max_len)

    return ds.map(tokenize, batched=True, remove_columns=["text"])


def _make_training_args(job_id: str, hp: HyperParams, n_samples: int) -> TrainingArguments:
    out_dir = Path(ServiceConfig.DATA_DIR) / "checkpoints" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_device = int(hp.batch_size)
    steps_per_epoch = max(n_samples // per_device, 1)
    actual_gas = int(hp.gradient_accumulation_steps or 1)
    total_steps = steps_per_epoch // actual_gas * hp.n_epochs
    warmup = int(total_steps * hp.warmup_ratio)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    return TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=hp.n_epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=ServiceConfig.FINE_TUNING_EVAL_BATCH_SIZE,
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


def _maybe_split(ds: Dataset, val_ratio: float) -> Optional[Dataset]:
    return None if val_ratio == 0 else ds.train_test_split(val_ratio, seed=42)["test"]


def _latest_checkpoint(job_id: str) -> Optional[str]:
    base = Path(ServiceConfig.DATA_DIR) / "checkpoints" / job_id
    if not base.exists():
        return None
    ckpts = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None
