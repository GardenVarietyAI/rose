#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm==0.30.2"]
# ///
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters with a base MLX model.")
    parser.add_argument("--model", required=True, help="Path to base MLX model.")
    parser.add_argument("--adapters", required=True, help="Path to adapters directory.")
    parser.add_argument("--checkpoint-metadata", required=True, help="Path to checkpoint metadata JSON.")
    parser.add_argument("--output", required=True, help="Output directory for fused model.")
    parser.add_argument("--dequantize", action="store_true", help="Fuse with --dequantize (recommended for GGUF).")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapters).expanduser().resolve()
    checkpoint_metadata_file = Path(args.checkpoint_metadata).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    checkpoint_metadata = json.loads(checkpoint_metadata_file.read_text(encoding="utf-8"))
    best_iteration: int = checkpoint_metadata["iteration"]
    checkpoint_file = f"{best_iteration:07d}_adapters.safetensors"

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Merging LoRA adapters with base model...")
    logger.info("Model: %s", model_dir)
    logger.info("Adapter: %s", adapter_dir)
    if checkpoint_file:
        logger.info("Checkpoint: %s", checkpoint_file)
    logger.info("Output: %s", output_dir)
    logger.info("Dequantize: %s", args.dequantize)

    _run_fuse(model_dir, adapter_dir, checkpoint_file, output_dir, args.dequantize)

    logger.info("Merged successfully.")
    logger.info("Output: %s", output_dir)


def _run_fuse(
    model_dir: Path,
    adapter_dir: Path,
    checkpoint_file: str | None,
    output_dir: Path,
    dequantize: bool,
) -> None:
    # mlx_lm.fuse expects adapters.safetensors, so create a temp symlink if needed
    if checkpoint_file and checkpoint_file != "adapters.safetensors":
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_file = adapter_dir / "adapter_config.json"
            if config_file.is_file():
                os.symlink(config_file, tmp_path / "adapter_config.json")
            os.symlink(adapter_dir / checkpoint_file, tmp_path / "adapters.safetensors")
            _run_fuse_cmd(model_dir, tmp_path, output_dir, dequantize)
            return
    _run_fuse_cmd(model_dir, adapter_dir, output_dir, dequantize)


def _run_fuse_cmd(
    model_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
    dequantize: bool,
) -> None:
    config: dict[str, str | int | float | bool | None] = {
        "--model": str(model_dir),
        "--adapter-path": str(adapter_dir),
        "--save-path": str(output_dir),
        "--dequantize": dequantize,
    }

    cmd: list[str] = [sys.executable, "-m", "mlx_lm", "fuse"]
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
