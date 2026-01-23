#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["mlx-lm==0.30.2"]
# ///
import argparse
import logging
from pathlib import Path

from mlx_lm import convert  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model to MLX format.")
    parser.add_argument("--model", required=True, help="Path to HF model directory or HF model ID.")
    parser.add_argument("--output", required=True, help="Output directory for MLX model.")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model (4-bit).")
    parser.add_argument("--q-bits", type=int, default=4, help="Quantization bits (default: 4).")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for non-quantized conversion.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s -> %s", model_path, output_dir)

    kwargs: dict[str, str | int | bool] = {
        "hf_path": str(model_path),
        "mlx_path": str(output_dir),
    }
    if args.quantize:
        kwargs["quantize"] = True
        kwargs["q_bits"] = args.q_bits
    elif args.dtype:
        kwargs["dtype"] = args.dtype
    convert(**kwargs)

    logger.info("Conversion complete: %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
