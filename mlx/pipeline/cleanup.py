#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove intermediate artifacts after successful run.")
    parser.add_argument("--base-model", required=True, help="Path to base MLX model directory.")
    parser.add_argument("--fused-model", required=True, help="Path to fused model directory.")
    parser.add_argument("--fp16-gguf", required=True, help="Path to fp16 GGUF file.")
    args = parser.parse_args()

    base_model = Path(args.base_model).expanduser().resolve()
    fused_model = Path(args.fused_model).expanduser().resolve()
    fp16_gguf = Path(args.fp16_gguf).expanduser().resolve()

    for path in [base_model, fused_model]:
        if path.is_dir():
            logger.info("Removing directory: %s", path)
            shutil.rmtree(path)

    if fp16_gguf.is_file():
        logger.info("Removing file: %s", fp16_gguf)
        fp16_gguf.unlink()

    logger.info("Cleanup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
