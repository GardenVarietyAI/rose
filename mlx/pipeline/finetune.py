#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm==0.30.2", "pyyaml"]
# ///
# pyright: reportMissingImports=false
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, train
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.utils import linear_to_lora_layers

import mlx.core as mlx_core
import mlx.optimizers as optim

logger = logging.getLogger(__name__)

OPTIMIZERS: dict[str, type[optim.Optimizer]] = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


class EarlyStopException(Exception):
    pass


class BestCheckpointTracker:
    def __init__(self, output_dir: Path, patience: int, min_delta: float) -> None:
        self.output_dir = output_dir
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss: float = float("inf")
        self.best_iteration: int = 1
        self.final_iteration: int = 1
        self.final_val_loss: float = float("inf")
        self.evals_without_improvement: int = 0

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        pass

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        iteration: int = val_info["iteration"]
        val_loss: float = val_info["val_loss"]
        checkpoint_iteration = iteration + 1

        self.final_iteration = checkpoint_iteration
        self.final_val_loss = val_loss

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_iteration = checkpoint_iteration
            self.evals_without_improvement = 0
            logger.info("New best at iter %d: val_loss=%.6f", checkpoint_iteration, val_loss)
        else:
            self.evals_without_improvement += 1
            logger.info(
                "No improvement at iter %d (val_loss=%.6f, best=%.6f, patience=%d/%d)",
                checkpoint_iteration,
                val_loss,
                self.best_val_loss,
                self.evals_without_improvement,
                self.patience,
            )
            if self.patience > 0 and self.evals_without_improvement >= self.patience:
                logger.info("Early stopping triggered at iter %d", checkpoint_iteration)
                raise EarlyStopException()

    def write_metadata(self, output_path: Path) -> None:
        result = {
            "iteration": self.best_iteration,
            "val_loss": self.best_val_loss,
            "final_iteration": self.final_iteration,
            "final_val_loss": self.final_val_loss,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote checkpoint metadata: %s", output_path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune an MLX model.")
    parser.add_argument("--model", required=True, help="Path to MLX model directory.")
    parser.add_argument("--train", required=True, help="Path to train.jsonl file.")
    parser.add_argument("--valid", required=True, help="Path to valid.jsonl file.")
    parser.add_argument("--output", required=True, help="Output directory for adapters.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint-metadata", required=True, help="Path to write checkpoint metadata JSON.")
    parser.add_argument("--patience", type=int, required=True, help="Early stopping patience (0 to disable).")
    parser.add_argument("--min-delta", type=float, required=True, help="Minimum improvement to reset patience.")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    train_file = Path(args.train).expanduser().resolve()
    valid_file = Path(args.valid).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_metadata_path = Path(args.checkpoint_metadata).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting LoRA fine-tuning...")
    logger.info("Model: %s", model_dir)
    logger.info("Train: %s", train_file)
    logger.info("Valid: %s", valid_file)
    logger.info("Output: %s", output_dir)

    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lora_params: dict[str, Any] = config["lora_parameters"]
    lr_config: dict[str, Any] = config["lr_schedule"]
    fine_tune_type: str = config["fine_tune_type"]
    optimizer_name: str = config["optimizer"]

    mlx_core.random.seed(config["seed"])

    model, tokenizer = load(str(model_dir))
    linear_to_lora_layers(model, config["num_layers"], lora_params)

    train_data = load_jsonl(train_file)
    valid_data = load_jsonl(valid_file)

    train_dataset = CacheDataset(ChatDataset(train_data, tokenizer, mask_prompt=config["mask_prompt"]))
    valid_dataset = CacheDataset(ChatDataset(valid_data, tokenizer, mask_prompt=config["mask_prompt"]))

    lr_args: list[float] = lr_config["arguments"]
    warmup_steps: int = lr_config["warmup"]
    warmup_init: float = float(lr_config["warmup_init"])
    peak_lr: float = float(lr_args[0])
    decay_steps: int = int(lr_args[1])
    final_lr: float = float(lr_args[2])

    warmup_schedule = optim.linear_schedule(
        init=warmup_init,
        end=peak_lr,
        steps=warmup_steps,
    )
    cosine_schedule = optim.cosine_decay(
        init=peak_lr,
        decay_steps=decay_steps - warmup_steps,
        end=final_lr,
    )
    lr_schedule = optim.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps],
    )
    optimizer = OPTIMIZERS[optimizer_name](learning_rate=lr_schedule)

    tracker = BestCheckpointTracker(output_dir, args.patience, args.min_delta)

    training_args = TrainingArgs(
        iters=config["iters"],
        batch_size=config["batch_size"],
        val_batches=config["val_batches"],
        steps_per_report=config["steps_per_report"],
        steps_per_eval=config["steps_per_eval"],
        steps_per_save=config["save_every"],
        max_seq_length=config["max_seq_length"],
        grad_checkpoint=config["grad_checkpoint"],
        grad_accumulation_steps=config["grad_accumulation_steps"],
        adapter_file=str(output_dir / "adapters.safetensors"),
    )

    try:
        train(
            model,
            optimizer,
            train_dataset,
            valid_dataset,
            args=training_args,
            training_callback=tracker,
        )
    except EarlyStopException:
        pass

    adapter_config = {
        "fine_tune_type": fine_tune_type,
        "num_layers": config["num_layers"],
        "lora_parameters": lora_params,
    }
    adapter_config_path = output_dir / "adapter_config.json"
    adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote adapter config: %s", adapter_config_path)

    tracker.write_metadata(checkpoint_metadata_path)

    logger.info("Fine-tuning complete.")
    logger.info("Adapters: %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
