"""Fine-tuning orchestrator that routes to appropriate trainer implementation."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

from rose_trainer.callbacks import CancellationCallback, EventCallback, HardwareMonitorCallback
from rose_trainer.client import ServiceClient
from rose_trainer.huggingface_trainer import (
    HuggingfaceTrainer,
)
from rose_trainer.types import Hyperparameters, LoraModelConfig, ModelConfig

logger = logging.getLogger(__name__)


def resolve_model_path(data_dir: str, model_id: str) -> Path:
    """Resolve model path and verify it exists."""
    model_path = Path(data_dir) / "models" / model_id
    if not model_path.exists():
        raise ValueError(f"Model path '{model_path}' does not exist")
    return model_path


def create_training_file_from_content(data_dir: str, training_file_id: str, content: bytes) -> Path:
    """Create temporary training file from content."""
    temp_dir = Path(data_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"{training_file_id}.jsonl"
    temp_file.write_bytes(content)
    logger.info(f"Created temporary training file: {temp_file}")
    return temp_file


def resolve_checkpoint_dir(data_dir: str, job_id: str) -> Path:
    """Resolve the checkpoint directory for a training job."""
    checkpoints_base = Path(data_dir) / "checkpoints"
    if not checkpoints_base.exists():
        raise ValueError(f"Checkpoints directory '{checkpoints_base}' does not exist")

    checkpoint_dir = checkpoints_base / job_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def generate_name(model_id: str, suffix: Optional[str] = None) -> str:
    """Generate a unique model name with timestamp."""
    ts = int(time.time())
    ft_model_id = f"{model_id}-ft-{ts}"
    if suffix:
        ft_model_id = f"{ft_model_id}-{suffix}"
    return ft_model_id


def create_output_dir(data_dir: str, model_id: str) -> Path:
    """Create output directory for fine-tuned model."""
    output_dir = Path(data_dir) / "models" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_training_time(seconds: float) -> str:
    """Format training time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.0f}h {minutes:.0f}m"


def get_optimal_device() -> str:
    """Get the optimal device for model inference/training."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_tokenizer(local_model_path: str) -> PreTrainedTokenizerBase:
    """Setup tokenizer with proper padding token."""
    logger.info(f"Loading tokenizer from local path: {local_model_path}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Log tokenizer config for debugging
    logger.info(f"Tokenizer config: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    return tokenizer


def get_model(
    model_info: ModelConfig,
    local_model_path: str,
    trust_remote_code: bool = True,  # Required for phi-2, Mistral, etc.
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    logger.info(f"Loading model from local path: {local_model_path}")
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            torch_dtype = torch.float32  # MPS doesn't support bfloat16
        else:
            torch_dtype = torch.bfloat16  # Better for CPU training

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
        local_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    logger.info(f"Successfully loaded model: {model_info.id}")
    return model


def apply_lora(model: PreTrainedModel, model_info: ModelConfig, lora_cfg: LoraModelConfig) -> PeftModel:
    """Apply LoRA configuration to model."""
    target_modules = (
        lora_cfg.target_modules or model_info.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    lora_model = get_peft_model(model, peft_config)  # type: ignore[arg-type]
    lora_model.print_trainable_parameters()
    return lora_model  # type: ignore[return-value]


def prepare_dataset(tokenizer: PreTrainedTokenizerBase, training_file_path: Path, max_length: int) -> Dataset:
    """Load and tokenize dataset."""
    dataset = load_dataset("json", data_files=str(training_file_path), split="train")

    def tokenize_function(samples: Dict[str, Any]) -> Dict[str, Any]:
        texts = []
        if "messages" in samples:
            for messages in samples["messages"]:
                if tokenizer.chat_template:
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                texts.append(text)
        elif "text" in samples:
            texts = samples["text"]
        elif "prompt" in samples:
            texts = samples["prompt"]
        else:
            raise ValueError("Dataset must contain 'messages', 'text', or 'prompt' fields.")

        result = tokenizer(texts, truncation=True, max_length=max_length)
        result["real_lengths"] = [len(ids) for ids in result["input_ids"]]
        return dict(result)

    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)


def save_model(
    model: Union[PreTrainedModel, PeftModel],
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    is_peft: bool,
) -> None:
    """Save fine-tuned model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_peft and hasattr(model, "merge_and_unload"):
        # Merge LoRA weights into base model for inference compatibility
        model = model.merge_and_unload()  # type: ignore

        # Make all tensors contiguous to avoid inference engine issues
        with torch.no_grad():
            for param in model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def train(
    job_id: str,
    model_name: str,
    training_file: str,
    hyperparameters: Hyperparameters,
    client: ServiceClient,
    event_callback: Callable[[str, str, Optional[Dict[str, Any]]], None],
    check_cancel_callback: Callable[[], str],
    config: Optional[Dict[str, Any]] = None,
    trainer: Optional[str] = "huggingface",
) -> Dict[str, Any]:
    """
    Main training entry point that routes to appropriate trainer.

    Args:
        job_id: Fine-tuning job ID
        model_name: Base model name
        hyperparameters: Training hyperparameters
        client: Service client for API communication
        check_cancel_callback: Function to check if job should be cancelled
        event_callback: Function to report training events
        config: Additional configuration (data_dir, checkpoint_dir, etc.)
        trainer: Trainer to use. Defaults to "huggingface"

    Returns:
        Dict containing training results
    """
    logger.info(f"Starting training with {trainer} trainer for job {job_id}")

    if trainer == "huggingface":
        # Fetch model info from API
        model_info: ModelConfig = client.get_model(model_name)

        # Set random seed for reproducibility
        set_random_seed(hyperparameters.seed)

        # Resolve paths and device
        if not config or "data_dir" not in config:
            raise ValueError("'data_dir' must be set")
        data_dir = config["data_dir"]

        local_model_path = resolve_model_path(data_dir, model_info.id)

        # Fetch training file content from API
        try:
            training_file_content = client.get_file_content(training_file)
        except FileNotFoundError:
            raise ValueError(f"Training file '{training_file}' not found")

        training_file_path = create_training_file_from_content(data_dir, training_file, training_file_content)
        ft_model_id = generate_name(model_info.id, hyperparameters.suffix)
        output_dir = create_output_dir(data_dir, ft_model_id)
        checkpoint_dir = resolve_checkpoint_dir(data_dir, job_id)
        device = get_optimal_device()

        # Load model and tokenizer
        event_callback("info", f"Loading model {model_info.id}", None)
        model = get_model(model_info, str(local_model_path))
        tokenizer = get_tokenizer(str(local_model_path))
        model = model.to(device)

        # Apply LoRA if enabled
        is_peft = False
        if hyperparameters.use_lora:
            lora_config = hyperparameters.lora_config or LoraModelConfig()
            model = apply_lora(model, model_info, lora_config)  # type: ignore[assignment]
            is_peft = True

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()  # Required when using gradient checkpointing
            logger.info("Enabled gradient checkpointing for memory efficiency")

        # Build training components
        hardware_monitor = HardwareMonitorCallback(event_callback)
        callbacks: List[TrainerCallback] = [
            EventCallback(event_callback),
            hardware_monitor,
            CancellationCallback(check_cancel_callback, job_id, checkpoint_dir=checkpoint_dir),
        ]
        if hyperparameters.validation_split > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=hyperparameters.early_stopping_patience))

        huggingface_trainer = HuggingfaceTrainer(
            model=model,
            tokenizer=tokenizer,
            is_peft=is_peft,
            hyperparams=hyperparameters,
            device=device,
            checkpoint_dir=checkpoint_dir,
            callbacks=callbacks,
        )
        try:
            # Prepare dataset
            event_callback("info", "Loading dataset", None)
            dataset = prepare_dataset(tokenizer, training_file_path, hyperparameters.max_length)

            # Run training
            huggingface_trainer.train(dataset)

            # Check if cancelled
            if check_cancel_callback() in ["cancelled", "cancelling"]:
                return {
                    "success": False,
                    "cancelled": True,
                    "steps": huggingface_trainer.global_step,
                    "tokens_processed": huggingface_trainer.total_tokens,
                }

            # Save model
            event_callback("info", f"Saving model to {output_dir}", None)
            save_model(model, tokenizer, output_dir, is_peft)

            # Get training results
            result = huggingface_trainer.save(output_dir, model_info.id, ft_model_id)

            # Log completion with time
            event_callback("info", f"Training completed in {format_training_time(result['training_time'])}", None)

            return cast(Dict[str, Any], result)

        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"HuggingFace training failed for job {job_id}: {str(e)}")
            raise
        finally:
            try:
                training_file_path.unlink()
                logger.info(f"Cleaned up temporary training file: {training_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary training file {training_file_path}: {e}")
    else:
        raise ValueError(f"Unknown trainer: {trainer}")
