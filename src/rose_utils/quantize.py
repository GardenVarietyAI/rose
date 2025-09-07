"""Quantize safetensors models using Candle tensor-tools."""

import subprocess
from pathlib import Path

import httpx
import typer
from rich import print


def quantize_model(
    in_path: Path,
    out_path: Path,
    quant: str = typer.Option("q4k", help="Quantization preset (e.g., q4k, q5k, q6k, q8_0)"),
    candle_root: Path = typer.Option("candle", help="Path to Candle repo"),
    register: bool = typer.Option(False, help="Register quantized model with API"),
    model_name: str = typer.Option(None, help="Model registry name for API registration"),
) -> None:
    """Quantize a safetensors model."""
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if register and not model_name:
        raise ValueError("model_name is required when register=True")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cargo",
        "run",
        "-p",
        "tensor-tools",
        "--release",
        "--",
        "quantize",
        "--quantization",
        quant,
        str(in_path),
        "--out-file",
        str(out_path.absolute()),
    ]
    print(f"[blue]Running quantization:[/blue] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=candle_root)
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError("Quantization failed!")

    print(f"[green]✓[/green] Quantized model saved to: {out_path}")

    if register:
        try:
            response = httpx.post(
                "http://localhost:8004/v1/models",
                json={
                    "model_name": model_name,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "timeout": 120,
                    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "quantized": True,
                    "quantization": quant,
                    "model_path": str(out_path.absolute()),
                },
            )
            response.raise_for_status()
            result = response.json()
            print(f"[green]✓[/green] Registered quantized: {result['id']}")

        except Exception as e:
            print(f"[red]Failed to register model: {e}[/red]")
