import json
from pathlib import Path

import torch
import typer
from rich.console import Console
from transformers import AutoModelForSequenceClassification, AutoTokenizer

console = Console()


def convert_reranker(
    model_path: Path = typer.Argument(..., help="Path to reranker model"),
    output_path: Path = typer.Argument(None, help="Output directory for ONNX model (defaults to {model_path}-ONNX)"),
) -> None:
    if not model_path.exists():
        console.print(f"[red]Model not found at {model_path}[/red]")
        raise typer.Exit(1)

    if output_path is None:
        output_path = Path(f"{model_path}-ONNX")

    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Loading model from {model_path}[/cyan]")

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model.eval()

    console.print("[green]Model loaded[/green]")

    # Create dummy input
    dummy_texts = ["query", "document"]
    inputs = tokenizer(dummy_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Export to ONNX
    onnx_path = output_path / "model.onnx"

    console.print("[cyan]Exporting to ONNX...[/cyan]")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs.input_ids, inputs.attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )

    console.print(f"[green]Exported to {onnx_path}[/green]")

    # Save tokenizer and config
    tokenizer.save_pretrained(output_path)
    model.config.save_pretrained(output_path)

    # Save metadata
    metadata = {
        "model_type": "ms_marco_minilm",
        "architecture": "cross_encoder",
        "num_labels": 1,
        "max_length": 512,
    }

    with open(output_path / "reranker_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print("[green]Conversion complete![/green]")
    console.print(f"[cyan]Output: {output_path}[/cyan]")
