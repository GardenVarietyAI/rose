from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def create_job(
    file_id: str = typer.Option(..., "--file", "-f", help="Training file ID"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Base model"),
    suffix: Optional[str] = typer.Option(None, "--suffix", help="Model suffix"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1.0, "--learning-rate", "-lr", help="Learning rate multiplier"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Create a fine-tuning job."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        console.print("[yellow]Creating fine-tuning job...[/yellow]")

        hyperparameters = {
            "n_epochs": epochs,
            "learning_rate_multiplier": learning_rate,
        }
        if batch_size:
            hyperparameters["batch_size"] = batch_size

        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model,
            suffix=suffix,
            hyperparameters=hyperparameters,
        )
        console.print(f"[green]Fine-tuning job created: {job.id}[/green]")
        console.print(f"Status: {job.status}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
