import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partialmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)

from ...config import ServiceConfig
from ...hf.loading import load_model_and_tokenizer
from ...model_registry import get_fine_tunable_models

logger = logging.getLogger(__name__)
@dataclass

class HyperParams:
    """User-supplied or auto-resolved hyper-parameters."""

    batch_size: int | str | None = None
    max_length: int | str | None = None
    n_epochs: int | str | None = 3
    learning_rate_multiplier: float | str | None = 1.0
    gradient_accumulation_steps: int | str | None = None
    validation_split: float = 0.1
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1
    use_lora: bool = True
    lora_config: dict | None = None
    seed: int = 42
    suffix: str = "custom"
    learning_rate: float = field(init=False)
    @classmethod

    def resolve(cls, raw: Dict, optimised=None) -> "HyperParams":
        """Coerce raw dict & 'auto' values into concrete numbers."""
        hp = cls(**raw)
        default_batch_size = 1
        default_max_length = 2048
        default_grad_accum = 4
        if optimised:
            hp.batch_size = cls._auto_int(hp.batch_size, optimised.batch_size)
            hp.max_length = cls._auto_int(hp.max_length, optimised.max_length)
            hp.gradient_accumulation_steps = cls._auto_int(
                hp.gradient_accumulation_steps, optimised.gradient_accumulation_steps
            )
        else:
            hp.batch_size = cls._auto_int(hp.batch_size, default_batch_size)
            hp.max_length = cls._auto_int(hp.max_length, default_max_length)
            hp.gradient_accumulation_steps = cls._auto_int(
                hp.gradient_accumulation_steps, default_grad_accum
            )
        hp.n_epochs = cls._auto_int(hp.n_epochs, 3)
        lr_mult = hp.learning_rate_multiplier
        if lr_mult is None or lr_mult == "auto":
            lr_mult = 1.0
        else:
            lr_mult = float(lr_mult)
        hp.learning_rate_multiplier = lr_mult
        hp.learning_rate = ServiceConfig.FINE_TUNING_DEFAULT_LEARNING_RATE * lr_mult
        return hp
    @staticmethod

    def _auto_int(val: int | str | None, fallback: int | None) -> int:
        if val is None or val == "auto":
            if fallback is None:
                return 1
            return int(fallback)
        return int(val)

class _BaseCallback(TrainerCallback):
    """Shared utilities for custom callbacks."""

    def __init__(self, event_cb: Optional[Callable] = None) -> None:
        self.event_cb = event_cb
        self._t0: float | None = None

    def _send(self, level: str, msg: str, data: Dict | None = None) -> None:
        if self.event_cb:
            self.event_cb(level, msg, data or {})

    def _eta(self, done: int, total: int) -> str:
        if not self._t0 or done == 0:
            return "--:--"
        seconds = (time.time() - self._t0) / done * (total - done)
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"

class HardwareMonitorCallback(_BaseCallback):
    """Simple hardware monitoring callback."""

    def on_log(self, args, state, control, logs=None, **_):
        if not logs or state.global_step % 10 != 0:
            return
        metrics = {}
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics["gpu_memory_used_gb"] = round(mem_info.used / 1024**3, 2)
                metrics["gpu_memory_total_gb"] = round(mem_info.total / 1024**3, 2)
                metrics["gpu_memory_percent"] = round(mem_info.used / mem_info.total * 100, 1)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["gpu_temperature"] = temp
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_utilization"] = util.gpu
            except Exception:
                pass
        elif torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / 1024**3
                metrics["mps_memory_gb"] = round(allocated, 2)
            except Exception:
                pass
        try:
            import psutil
            process = psutil.Process()
            metrics["cpu_percent"] = process.cpu_percent()
            metrics["ram_gb"] = round(process.memory_info().rss / 1024**3, 2)
        except Exception:
            pass
        if metrics:
            self._send("debug", "Hardware metrics", metrics)

class EventCallback(_BaseCallback):
    """Streams high-level training progress to ``event_callback``."""

    def on_train_begin(self, args, state, control, **_):
        self._t0 = time.time()

    def on_epoch_begin(self, args, state, control, **_):
        self._send(
            "info",
            f"Epoch {state.epoch + 1}/{args.num_train_epochs} started",
            {"epoch": state.epoch + 1},
        )

    def on_log(self, args, state, control, logs=None, **_):
        if not logs or "loss" not in logs:
            return
        total = args.max_steps
        if total <= 0:
            if hasattr(state, 'num_train_epochs') and hasattr(args, 'num_train_epochs'):
                if state.global_step > 0 and state.epoch > 0:
                    steps_per_epoch = int(state.global_step / state.epoch)
                    total = steps_per_epoch * args.num_train_epochs
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

class CancellationCallback(TrainerCallback):
    """Stops training early when an external controller requests it."""

    def __init__(
        self,
        status_fn: Callable[[], str],
        job_id: str,
    ) -> None:
        self._status_fn = status_fn
        self._job_id = job_id
        self.cancelled = False
        self.paused = False

    def on_step_end(self, args, state, control, **_) -> "TrainerControl":
        match self._status_fn():
            case "cancelling":
                self.cancelled, control.should_training_stop = True, True
            case "pausing":
                self.paused, control.should_save, control.should_training_stop = True, True, True
        return control

    def on_save(self, args, state, control, **kwargs):
        if self.paused:
            meta = {
                "is_paused": True,
                "global_step": state.global_step,
                "epoch": state.epoch,
            }
            (Path(ServiceConfig.DATA_DIR) / "checkpoints" / self._job_id / "pause_meta.json").write_text(
                json.dumps(meta)
            )
        return control

class HFTrainer:
    """Wraps HF Trainer with checkpoint management and resource monitoring."""

    def __init__(self) -> None:
        self.fine_tuning_models = get_fine_tunable_models()

    def train(
        self,
        job_id: str,
        model_name: str,
        training_file_path: Path,
        hyperparameters: Dict,
        check_cancel_callback: Optional[Callable[[], str]] = None,
        event_callback: Optional[Callable[[str, str, Dict], None]] = None,
    ) -> Dict:
        """Run a fine-tuning job and return a result dict."""
        _silence_transformers()
        raw_data = _load_jsonl(training_file_path)
        hp = HyperParams.resolve(hyperparameters)
        torch.manual_seed(hp.seed)
        np.random.seed(hp.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(hp.seed)
        model, tok = self._load_model_and_tok(model_name, hp)
        ds = _prepare_dataset(raw_data, tok, hp.max_length)
        args = _make_training_args(job_id, hp, len(ds))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)
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
            processing_class=tok,
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
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                    },
                )
            result = trainer.train(resume_from_checkpoint=_latest_checkpoint(job_id))
            out_dir = self._save_model(trainer, model_name, hp.suffix)
            metrics = result.metrics
            tokens = int(metrics.get("train_samples_per_second", 0) * trainer.state.global_step)
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

    def _load_model_and_tok(self, model_name: str, hp: HyperParams):
        """Load model with hyperparameters applied."""

        if model_name not in self.fine_tuning_models:
            raise ValueError(f"Model {model_name} not supported for fine-tuning")
        hf_model_name = self.fine_tuning_models[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = load_model_and_tokenizer(
            model_id=hf_model_name,
            device=device,
            device_map=None,
            offload_dir=None,
        )
        if hp.use_lora:
            lora_cfg = hp.lora_config or {}
            lora_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                target_modules=lora_cfg.get("target_modules"),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        return model, tokenizer

    def _save_model(self, trainer: Trainer, base_name: str, suffix: str) -> Path:
        ts = int(time.time())
        model_id = f"{base_name}-ft-{ts}-{suffix}"
        out = Path(ServiceConfig.DATA_DIR) / "models" / model_id
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        if hasattr(trainer.model, "merge_and_unload"):
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
        self._update_registry(model_id, str(out), base_name)
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
        registry[model_id] = {
            "path": model_path,
            "base_model": base_model,
            "created_at": time.time()
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

def _load_jsonl(fp: Path) -> List[Dict]:
    """Read JSONL file, ignoring blank lines."""
    lines = [ln for ln in fp.read_text("utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]

def _prepare_dataset(raw: Sequence[Dict], tok, max_len: int) -> Dataset:

    def to_text(item: Dict) -> str:
        if "messages" in item and hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        if "prompt" in item and "completion" in item:
            return item["prompt"] + item["completion"]
        return item.get("text", "")
    texts = [to_text(ex) for ex in raw]
    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)
    return ds.map(tokenize, batched=True, remove_columns=["text"])

def _make_training_args(job_id: str, hp: HyperParams, n_samples: int) -> TrainingArguments:
    out_dir = Path(ServiceConfig.DATA_DIR) / "checkpoints" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_device = int(hp.batch_size)
    steps_per_epoch = max(n_samples // per_device, 1)
    actual_gas = int(hp.gradient_accumulation_steps or 1)
    total_steps = steps_per_epoch // actual_gas * hp.n_epochs
    warmup = int(total_steps * hp.warmup_ratio)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=hp.n_epochs,
        per_device_train_batch_size=per_device,
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
    )

def _maybe_split(ds: Dataset, val_ratio: float) -> Optional[Dataset]:
    return None if val_ratio == 0 else ds.train_test_split(val_ratio, seed=42)["test"]

def _latest_checkpoint(job_id: str) -> Optional[str]:
    base = Path(ServiceConfig.DATA_DIR) / "checkpoints" / job_id
    if not base.exists():
        return None
    ckpts = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None

def _silence_transformers() -> None:
    os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
    from tqdm import tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)