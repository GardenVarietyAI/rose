import subprocess
from pathlib import Path

import typer

app = typer.Typer()


def quantize(
    in_path: Path,
    out_path: Path,
    quant: str = typer.Option("q4_k_m", help="Quantization preset (e.g., q4_k_m, q5_k_m, q6k, q8_0)"),
    candle_root: Path = typer.Option("candle", help="Path to Candle repo"),
) -> None:
    """
    Quantize a safetensors model using Candle's tensor-tools.
    """
    assert in_path.exists(), f"{in_path} not found"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    cmd = [
        "cargo",
        "run",
        "-p",
        "candle-examples",
        "--example",
        "tensor-tools",
        "--release",
        "--",
        "quantize",
        "--quantization",
        quant,
        str(in_path),
        "--out-file",
        str(out_path),
    ]
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=candle_root)
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Quantization failed!")
