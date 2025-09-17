from pathlib import Path

import typer
from optimum.onnxruntime import ORTModelForSequenceClassification
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoConfig, AutoTokenizer

console = Console()


def convert_reranker(
    model_path: Path = typer.Argument(..., help="Path to reranker model"),
    output_path: Path = typer.Argument(..., help="Output directory for ONNX model"),
) -> None:
    """Convert a Qwen reranker model to ONNX format for CPU inference."""

    if not model_path.exists():
        console.print(f"[red]Model path does not exist: {model_path}[/red]")
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading model configuration...", total=None)

        try:
            config = AutoConfig.from_pretrained(model_path)
            console.print(f"[green]✓ Loaded model config from {model_path}[/green]")
            console.print(f"[cyan]Model type: {config.model_type}[/cyan]")
        except Exception as e:
            console.print(f"[red]Failed to load model config: {e}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        console.print(f"[green]✓ Loaded tokenizer with {len(tokenizer)} tokens[/green]")

        progress.update(task, description="Converting to ONNX (this may take a few minutes)...")

        try:
            # Export as sequence classification model (reranker outputs scores)
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_path,
                export=True,
                trust_remote_code=True,
            )
            ort_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            console.print("[green]✓ Converted model to ONNX format[/green]")
        except Exception as e:
            console.print(f"[red]Failed to convert to ONNX: {e}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="Creating reranker configuration...")
        progress.remove_task(task)

    console.print("[green]✓ Reranker model converted successfully![/green]")
    console.print(f"[cyan]Output directory: {output_path}[/cyan]")


if __name__ == "__main__":
    typer.run(convert_reranker)
