#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm>=0.30.2"]
# ///
import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune an MLX model.")
    parser.add_argument("--model", required=True, help="Path to MLX model directory.")
    parser.add_argument("--train", required=True, help="Path to train.jsonl file.")
    parser.add_argument("--valid", required=True, help="Path to valid.jsonl file.")
    parser.add_argument("--output", required=True, help="Output directory for adapters.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--log-file", help="Path to write training log output.")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    train_file = Path(args.train).expanduser().resolve()
    valid_file = Path(args.valid).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting LoRA fine-tuning...")
    logger.info("Model: %s", model_dir)
    logger.info("Train: %s", train_file)
    logger.info("Valid: %s", valid_file)
    logger.info("Output: %s", output_dir)

    with tempfile.TemporaryDirectory(prefix=".data-", dir=str(output_dir)) as tmp_dir:
        temp_data_dir = Path(tmp_dir)
        (temp_data_dir / "train.jsonl").symlink_to(train_file.absolute())
        (temp_data_dir / "valid.jsonl").symlink_to(valid_file.absolute())

        config_path = Path(args.config).expanduser().resolve()

        config: dict[str, Any] = {
            "--model": str(model_dir),
            "--train": True,
            "--data": str(temp_data_dir),
            "--adapter-path": str(output_dir),
            "--config": str(config_path),
        }

        cmd: list[str] = [sys.executable, "-m", "mlx_lm", "lora"]
        for flag, value in config.items():
            if value is None or value is False:
                continue
            if value is True:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        log_file_handle = None
        if args.log_file:
            log_path = Path(args.log_file).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_handle = log_path.open("w", encoding="utf-8")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if log_file_handle:
                    log_file_handle.write(line)
                    log_file_handle.flush()
            returncode = proc.wait()
            if returncode != 0:
                cmd_str = " ".join(cmd)
                logger.error("Command failed (%s): %s", returncode, cmd_str)
                sys.exit(returncode)
        finally:
            if log_file_handle:
                log_file_handle.close()

    logger.info("Fine-tuning complete.")
    logger.info("Adapters: %s", output_dir)
    if args.log_file:
        logger.info("Log: %s", args.log_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
