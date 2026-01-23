#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["lm-eval==0.4.9.2", "transformers==5.0.0rc1"]
# ///
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace model on benchmarks using lm-eval.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path.")
    parser.add_argument("--tasks", required=True, help="Comma-separated list of lm-eval tasks.")
    parser.add_argument("--output", required=True, help="Path to write evaluation results JSON.")
    parser.add_argument("--limit", type=int, help="Limit number of examples per task (for testing).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1).")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating HF model: %s", args.model)
    logger.info("Tasks: %s", args.tasks)

    config: dict[str, Any] = {
        "--model": "hf",
        "--model_args": f"pretrained={args.model},device=mps",
        "--tasks": args.tasks,
        "--batch_size": args.batch_size,
        "--output_path": str(output_path),
        "--limit": args.limit,
    }

    cmd: list[str] = [sys.executable, "-m", "lm_eval"]
    for flag, value in config.items():
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("lm_eval failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("Results written to: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
