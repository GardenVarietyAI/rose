#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

VAL_LOSS_PATTERN = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def parse_training_log(log_path: Path) -> list[tuple[int, float]]:
    results: list[tuple[int, float]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = VAL_LOSS_PATTERN.search(line)
            if match:
                iteration = int(match.group(1))
                val_loss = float(match.group(2))
                results.append((iteration, val_loss))
    return results


def find_checkpoint_file(adapter_dir: Path, iteration: int) -> Path | None:
    patterns = [
        f"{iteration:07d}_adapters.safetensors",
        f"{iteration:06d}_adapters.safetensors",
        f"{iteration:05d}_adapters.safetensors",
        f"{iteration}_adapters.safetensors",
        f"adapters-{iteration:05d}.safetensors",
        f"adapters-{iteration:04d}.safetensors",
        f"adapters-{iteration}.safetensors",
    ]
    for pattern in patterns:
        candidate = adapter_dir / pattern
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best checkpoint by validation loss.")
    parser.add_argument("--log", required=True, help="Path to training log file.")
    parser.add_argument("--adapter-dir", required=True, help="Path to adapter directory with checkpoints.")
    parser.add_argument("--output", help="Write best checkpoint path to file (optional).")
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()

    results = parse_training_log(log_path)

    best_iter, best_loss = min(results, key=lambda x: x[1])
    final_iter, final_loss = results[-1]

    logger.info("Found %d validation checkpoints", len(results))
    logger.info("Best: iter %d, val_loss %.6f", best_iter, best_loss)
    logger.info("Final: iter %d, val_loss %.6f", final_iter, final_loss)

    checkpoint_file = find_checkpoint_file(adapter_dir, best_iter)
    final_adapter = adapter_dir / "adapters.safetensors"

    if checkpoint_file:
        logger.info("Best checkpoint file: %s", checkpoint_file)
        result_path = checkpoint_file
    elif best_iter == final_iter and final_adapter.is_file():
        logger.info("Best is final checkpoint: %s", final_adapter)
        result_path = final_adapter
    else:
        logger.warning("Checkpoint file for iter %d not found, using final adapter", best_iter)
        result_path = final_adapter

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(str(result_path) + "\n", encoding="utf-8")
        logger.info("Wrote best checkpoint path to: %s", output_path)

    print(result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    main()
