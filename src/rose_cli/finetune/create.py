from typing import Any, Optional

import typer

from rose_cli.utils import console, get_client


def create_job(
    file_id: str = typer.Option(..., "--file", "-f", help="Training file ID"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Base model"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Model suffix"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    learning_rate_multiplier: float = typer.Option(
        1.0, "--learning-rate-multiplier", "-lrm", help="Learning rate multiplier (e.g., 1.0, 2.0, 4.0)"
    ),
    validation_split: Optional[float] = typer.Option(
        None, "--validation-split", "-vs", help="Validation split ratio (0.0-1.0)"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the job ID"),
) -> None:
    """Create a fine-tuning job."""
    client = get_client()
    try:
        if not quiet:
            console.print("[yellow]Creating fine-tuning job...[/yellow]")

        hyperparameters: dict[str, Any] = {
            "n_epochs": epochs,
            "learning_rate_multiplier": learning_rate_multiplier,
        }
        if batch_size is not None:
            hyperparameters["batch_size"] = batch_size
        if validation_split is not None:
            hyperparameters["validation_split"] = validation_split

        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model,
            suffix=suffix,
            hyperparameters=hyperparameters,
        )

        if quiet:
            print(job.id)
        else:
            console.print(f"[green]Fine-tuning job created: {job.id}[/green]")
            console.print(f"Status: {job.status}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
