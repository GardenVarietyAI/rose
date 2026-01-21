#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm>=0.30.2", "datasets"]
# ///
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on a dataset.")
    parser.add_argument("--model", required=True, help="Path to MLX model directory.")
    parser.add_argument("--adapter", help="Path to adapter directory.")
    parser.add_argument("--data", required=True, help="HuggingFace dataset name or path.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument("--sequence-length", type=int, default=2048, help="Sequence length (default: 2048).")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples, -1 for all (default: -1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve() if args.adapter else None

    logger.info("Evaluating perplexity...")
    logger.info("Model: %s", model_dir)
    if adapter_dir:
        logger.info("Adapter: %s", adapter_dir)
    logger.info("Data: %s", args.data)

    if adapter_dir:
        _run_with_adapter(model_dir, adapter_dir, args)
    else:
        _run_perplexity(model_dir, args)


def _run_with_adapter(model_dir: Path, adapter_dir: Path, args: argparse.Namespace) -> None:
    # mlx_lm perplexity doesn't support --adapter-path, use lora --test instead
    config: dict[str, str | int | float | bool | None] = {
        "--model": str(model_dir),
        "--adapter-path": str(adapter_dir),
        "--test": True,
        "--test-batches": args.num_samples if args.num_samples > 0 else -1,
        "--max-seq-length": args.sequence_length,
        "--batch-size": args.batch_size,
        "--data": args.data,
        "--seed": args.seed,
    }

    cmd: list[str] = [sys.executable, "-m", "mlx_lm", "lora"]
    for flag, value in config.items():
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)


def _run_perplexity(model_dir: Path, args: argparse.Namespace) -> None:
    config: dict[str, str | int | float | bool | None] = {
        "--model": str(model_dir),
        "--data-path": args.data,
        "--batch-size": args.batch_size,
        "--sequence-length": args.sequence_length,
        "--num-samples": args.num_samples,
        "--seed": args.seed,
    }

    cmd: list[str] = [sys.executable, "-m", "mlx_lm", "perplexity"]
    for flag, value in config.items():
        if value is None:
            continue
        cmd.extend([flag, str(value)])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
