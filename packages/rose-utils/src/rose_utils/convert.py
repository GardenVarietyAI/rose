"""Convert sentence-transformers models to ONNX format."""

import json
from pathlib import Path

import typer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

console = Console()


def convert_model(
    model_path: Path = typer.Argument(..., help="Path to sentence-transformers model"),
    output_path: Path = typer.Argument(..., help="Output directory for ONNX model"),
) -> None:
    """Convert a sentence-transformers model to ONNX format with LAST_TOKEN pooling."""

    if not model_path.exists():
        console.print(f"[red]Model path does not exist: {model_path}[/red]")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading sentence-transformers model...", total=None)

        try:
            st_model = SentenceTransformer(str(model_path))
            console.print(f"[green]✓ Loaded sentence-transformers model from {model_path}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load sentence-transformers model: {e}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Loading transformers model...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]

        progress.update(task, description="Converting to ONNX...")

        try:
            ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True)
            ort_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
        except Exception as e:
            console.print(f"[red]Failed to convert to ONNX: {e}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Creating FastEmbed configuration...")

        config = {
            "model_name": model_path.name,
            "pooling": "LAST_TOKEN",
            "normalization": True,
            "dimensions": st_model.get_sentence_embedding_dimension(),
            "description": "Converted from sentence-transformers with LAST_TOKEN pooling",
        }

        config_path = output_path / "fastembed_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        progress.remove_task(task)

    console.print("[green]✅ Model converted successfully![/green]")
    console.print(f"[cyan]Output directory: {output_path}[/cyan]")
    console.print(f"[cyan]Embedding dimensions: {st_model.get_sentence_embedding_dimension()}[/cyan]")
