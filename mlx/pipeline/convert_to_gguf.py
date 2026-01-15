#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize fp16 GGUF to target format.")
    parser.add_argument("--input", required=True, help="Path to fp16 GGUF file.")
    parser.add_argument("--output", required=True, help="Output path for quantized GGUF file.")
    parser.add_argument("--quant", default="Q4_K_M", help="Quantization type for llama-quantize (default: Q4_K_M).")
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

    metadata = {
        "created_at_unix": int(time.time()),
        "git_commit": _git_commit(input_path.parent),
        "llama_cpp_commit": _git_commit(llama_cpp_dir),
        "input_fp16": str(input_path),
        "llama_cpp_dir": str(llama_cpp_dir),
        "quant": args.quant,
        "output": str(output_path),
    }
    Path(str(output_path) + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
