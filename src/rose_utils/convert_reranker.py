"""Convert Qwen reranker models to ONNX format."""

import json
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

        # Create reranker-specific config
        reranker_config = {
            "model_name": model_path.name,
            "model_type": "reranker",
            "scoring_method": "logits",
            "input_format": "query_document_pairs",
            "max_length": getattr(config, "max_position_embeddings", 32768),
            "num_labels": getattr(config, "num_labels", 2),
            "description": "Qwen reranker model for scoring query-document relevance",
            "instruction_template": "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
            "default_instruction": "Given a web search query, retrieve relevant passages that answer the query",
        }

        config_path = output_path / "reranker_config.json"
        config_path.write_text(json.dumps(reranker_config, indent=2))

        # Also save ONNX optimization config for CPU
        optimization_config = {
            "optimization_level": 99,  # Enable all optimizations
            "optimize_for_gpu": False,  # CPU only
            "fp16": False,  # Keep fp32 for CPU
            "enable_transformers_specific_optimizations": True,
        }

        opt_config_path = output_path / "onnx_optimization.json"
        opt_config_path.write_text(json.dumps(optimization_config, indent=2))

        progress.remove_task(task)

    console.print("[green]✅ Reranker model converted successfully![/green]")
    console.print(f"[cyan]Output directory: {output_path}[/cyan]")
    console.print(f"[cyan]Max sequence length: {reranker_config['max_length']}[/cyan]")
    console.print(f"[cyan]Number of labels: {reranker_config['num_labels']}[/cyan]")


if __name__ == "__main__":
    typer.run(convert_reranker)
