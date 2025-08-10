import os
from pathlib import Path

import typer
from huggingface_hub import HfFolder, snapshot_download

from rose_cli.utils import console, get_client


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
    alias: str = typer.Option(None, "--alias", "-a", help="Short alias for the model (defaults to last part of name)"),
    num_workers: int = typer.Option(1, "--num-workers", "-n", help="Number of files to download simulatneously"),
) -> None:
    """Download a model from HuggingFace and register it in the database."""
    # model_name is the HuggingFace model ID
    hf_model_name = model_name

    # Determine local path
    models_dir = get_models_directory()
    safe_model_name = hf_model_name.replace("--", "/")
    local_dir = models_dir / safe_model_name

    # Check if already exists
    if local_dir.exists() and not force:
        console.print(f"[yellow]Model {model_name} already downloaded at {local_dir}[/yellow]")
        console.print("[dim]Use --force to re-download[/dim]")
        return

    try:
        console.print(f"[yellow]Downloading {hf_model_name} to {local_dir}[/yellow]")
        console.print("[dim]This may take several minutes for large models...[/dim]\n")

        # Download using snapshot_download
        local_path = snapshot_download(
            repo_id=hf_model_name,
            local_dir=str(local_dir),
            force_download=force,
            token=HfFolder.get_token(),
            max_workers=num_workers,
        )

        console.print(f"[green]✓ Model {model_name} successfully downloaded[/green]")
        console.print(f"[dim]Path: {local_path}[/dim]")

        # Register model in database
        client = get_client()

        # Use alias if provided, otherwise use the full model name
        model_id = alias if alias else hf_model_name

        # Build auth headers
        headers = {}
        if client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"

        try:
            # Register the model
            response = client._client.post(
                "/models",
                json={
                    "id": model_id,
                    "model_name": hf_model_name,
                    "name": hf_model_name.split("/")[-1],
                    "owned_by": hf_model_name.split("/")[0].lower(),
                },
                headers=headers,
            )
            response.raise_for_status()
            console.print(f"[green]✓ Model registered as '{model_id}'[/green]")
        except Exception as e:
            if "already exists" in str(e):
                console.print(f"[yellow]Model '{model_id}' already registered[/yellow]")
            else:
                console.print(f"[yellow]Warning: Failed to register model: {e}[/yellow]")
                console.print("[dim]You can manually register it with: rose models add[/dim]")

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        raise typer.Exit(1)
