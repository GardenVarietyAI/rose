"""Quantize safetensors models using Candle tensor-tools."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich import print


def quantize_model(
    in_path: Path,
    out_path: Path,
    quant: str = typer.Option("q4k", help="Quantization preset (e.g., q4k, q5k, q6k, q8_0)"),
    candle_root: Path = typer.Option("candle", help="Path to Candle repo"),
    register: bool = typer.Option(False, help="Register quantized model with API"),
    model_name: Optional[str] = typer.Option(None, help="Model registry name for API registration"),
    parent_model: Optional[str] = typer.Option(None, help="Parent model name for fine-tuned models"),
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
            # Models need to be in directories, not direct files
            # Create a directory if the output is a direct file
            if out_path.suffix == ".gguf":
                model_dir = out_path.parent / out_path.stem
                model_dir.mkdir(exist_ok=True)
                # Move the GGUF file into the directory as "model.gguf"
                final_gguf_path = model_dir / "model.gguf"
                if out_path != final_gguf_path:
                    out_path.rename(final_gguf_path)

                # Copy tokenizer.json from the source safetensors directory
                source_dir = in_path.parent
                source_tokenizer = source_dir / "tokenizer.json"
                if source_tokenizer.exists():
                    shutil.copy2(source_tokenizer, model_dir / "tokenizer.json")
                    print(f"[blue]Copied tokenizer.json from {source_dir}[/blue]")

                model_path = f"models/{model_dir.name}"
                model_name_to_register = model_dir.name
            else:
                # Already in a directory structure
                model_path = f"models/{out_path.parent.name}"
                model_name_to_register = out_path.parent.name

            # Check if this is a fine-tuned model by looking for the pattern in model_name
            is_fine_tuned = "ft-" in str(model_name_to_register) or "dev-commands" in str(model_name_to_register)
            # Use provided parent_model or default to Qwen3-1.7B for fine-tuned models
            if parent_model is None and is_fine_tuned:
                parent_model = "Qwen--Qwen3-1.7B"

            json_payload = {
                "model_name": model_name_to_register,
                "path": model_path,
                "kind": "qwen3_gguf",
                "temperature": 0.7,
                "top_p": 0.9,
                "timeout": 120,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "quantization": quant,
            }

            if parent_model:
                json_payload["parent"] = parent_model

            response = httpx.post(
                "http://localhost:8004/v1/models",
                json=json_payload,
            )
            response.raise_for_status()
            result = response.json()
            print(f"[green]✓[/green] Registered quantized: {result['id']}")

        except Exception as e:
            print(f"[red]Failed to register model: {e}[/red]")
