#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm>=0.30.2"]
# ///
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model to MLX format.")
    parser.add_argument("--model", required=True, help="Path to HF model directory or HF model ID.")
    parser.add_argument("--output", required=True, help="Output directory for MLX model.")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model (4-bit).")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits (default: 4).")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], help="Data type for non-quantized conversion.")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [sys.executable, "-m", "mlx_lm", "convert"]
    cmd.extend(["--hf-path", str(model_path)])
    cmd.extend(["--mlx-path", str(output_dir)])

    if args.quantize:
        cmd.append("-q")
        cmd.extend(["--q-bits", str(args.q_bits)])
    elif args.dtype:
        cmd.extend(["--dtype", args.dtype])

    print(f"Converting {model_path} -> {output_dir}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Conversion failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Conversion complete: {output_dir}")


if __name__ == "__main__":
    main()
