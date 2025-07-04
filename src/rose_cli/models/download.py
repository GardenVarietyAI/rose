import os
from pathlib import Path

import typer
from huggingface_hub import snapshot_download
from rich.progress import Progress, SpinnerColumn, TextColumn

from rose_cli.utils import console


def get_models_directory() -> Path:
    """Get the local directory for storing downloaded models."""
    # Use same path as server expects
    data_dir = os.environ.get("ROSE_SERVER_DATA_DIR", "./data")
    models_dir = Path(data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_model(
    model_name: str = typer.Argument(..., help="HuggingFace model to download (e.g. microsoft/phi-2)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if exists"),
) -> None:
    """Download a model directly from HuggingFace."""
    # model_name is the HuggingFace model ID
    hf_model_name = model_name

    # Determine local path
    models_dir = get_models_directory()
    safe_model_name = hf_model_name.replace("/", "--")
    local_dir = models_dir / safe_model_name

    # Check if already exists
    if local_dir.exists() and not force:
        console.print(f"[yellow]Model {model_name} already downloaded at {local_dir}[/yellow]")
        console.print("[dim]Use --force to re-download[/dim]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {hf_model_name}...", total=None)

            # Download using snapshot_download
            local_path = snapshot_download(
                repo_id=hf_model_name,
                local_dir=str(local_dir),
                force_download=force,
                token=None,  # Public models only for now
            )

            progress.update(task, completed=True)

        console.print(f"[green]✓ Model {model_name} successfully downloaded[/green]")
        console.print(f"[dim]Path: {local_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        raise typer.Exit(1)
