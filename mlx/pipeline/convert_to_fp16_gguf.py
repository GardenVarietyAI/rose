#!/usr/bin/env -S uv run --script --prerelease=allow
# /// script
# dependencies = ["numpy==2.4.1", "sentencepiece==0.2.1", "transformers==5.0.0rc1", "protobuf==6.33.4", "torch==2.9.1"]
# ///
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF model to fp16 GGUF.")
    parser.add_argument("--model", required=True, help="HF model path.")
    parser.add_argument("--output", required=True, help="Output path for fp16 GGUF file.")
    parser.add_argument("--llama-cpp", default="vendor/llama.cpp", help="llama.cpp path (default: vendor/llama.cpp).")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    llama_cpp_dir = Path(args.llama_cpp).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    logger.info("Converting to fp16 GGUF...")
    logger.info("Input: %s", model_dir)
    logger.info("Output: %s", output_path)

    convert_cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(output_path),
        "--outtype",
        "f16",
    ]

    try:
        subprocess.run(convert_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)

    logger.info("Conversion complete: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
