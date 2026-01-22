#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize fp16 GGUF to target format.")
    parser.add_argument("--input", required=True, help="Path to fp16 GGUF file.")
    parser.add_argument("--output", required=True, help="Output path for quantized GGUF file.")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization type (default: Q4_K_M).")
    parser.add_argument("--llama-cpp", default="vendor/llama.cpp", help="Path to llama.cpp directory.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    llama_cpp_dir = Path(args.llama_cpp).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llama_quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    logger.info("Quantizing to %s...", args.quant)
    logger.info("Input: %s", input_path)
    logger.info("Output: %s", output_path)

    try:
        subprocess.run([str(llama_quantize_bin), str(input_path), str(output_path), args.quant], check=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)

    logger.info("Quantization complete: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
