#!/usr/bin/env python3

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build llama.cpp with Metal support.")
    parser.add_argument(
        "--llama-cpp",
        default="vendor/llama.cpp",
        help="Path to llama.cpp directory (default: vendor/llama.cpp).",
    )
    args = parser.parse_args()

    llama_cpp_dir = Path(args.llama_cpp).expanduser().resolve()

    if not llama_cpp_dir.is_dir():
        logger.error("llama.cpp directory not found: %s", llama_cpp_dir)
        logger.error("Hint: run: git submodule update --init --recursive")
        sys.exit(2)

    if shutil.which("cmake") is None:
        logger.error("Required command not found: cmake")
        sys.exit(127)

    logger.info("Building llama.cpp (Metal enabled)...")
    logger.info("Source: %s", llama_cpp_dir)

    cmake_configure_cmd = [
        "cmake",
        "-S",
        str(llama_cpp_dir),
        "-B",
        str(llama_cpp_dir / "build"),
        "-DGGML_METAL=ON",
    ]
    try:
        subprocess.run(cmake_configure_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)

    cmake_build_cmd = [
        "cmake",
        "--build",
        str(llama_cpp_dir / "build"),
        "--config",
        "Release",
        "-j",
    ]
    try:
        subprocess.run(cmake_build_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(str(part) for part in exc.cmd)
        logger.error("Command failed (%s): %s", exc.returncode, cmd_str)
        sys.exit(exc.returncode)

    logger.info("Build complete.")
    logger.info("Binaries: %s", llama_cpp_dir / "build" / "bin")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
