import os
from pathlib import Path

import typer
from huggingface_hub import HfFolder, hf_hub_download, snapshot_download

from rose_cli.utils import console, get_client

blessed_models = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-GGUF",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-0.6B-GGUF",
    "Qwen/Qwen3-Embedding-0.6B",
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
]


def _get_gguf_config(model_name: str) -> dict[str, str] | None:
    """Get GGUF file and base model config for blessed models."""
    gguf_configs = {
        "Qwen/Qwen3-0.6B-GGUF": {
            "base_model": "Qwen/Qwen3-0.6B",
            "gguf_file": "Qwen3-0.6B-Q8_0.gguf",
            "tokenizer_file": "tokenizer.json",
        },
        "Qwen/Qwen3-4B-GGUF": {
            "base_model": "Qwen/Qwen3-4B",
            "gguf_file": "Qwen3-4B-Q4_K_M.gguf",
            "tokenizer_file": "tokenizer.json",
        },
        "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF": {
            "base_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "gguf_file": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            "tokenizer_file": "tokenizer.json",
        },
    }
    return gguf_configs.get(model_name)


def _download_base_tokenizer(base_model_name: str, tokenizer_file: str, target_dir: Path, force: bool) -> None:
    """Download tokenizer from base model to target directory."""
    if (target_dir / tokenizer_file).exists() and not force:
        return

    try:
        hf_hub_download(
            repo_id=base_model_name,
            filename="tokenizer.json",
            local_dir=str(target_dir),
            token=HfFolder.get_token(),
        )
    except Exception as e:
        console.print(f"[red]Failed to download tokenizer: {e}[/red]")


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
) -> None:
    """Download a model from HuggingFace and register it in the database."""
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
        gguf_config = _get_gguf_config(hf_model_name)
        if not (gguf_config or hf_model_name in blessed_models):
            console.print(f"[red]Model '{hf_model_name}' is not supported.[/red]")
            console.print("[yellow]Supported models:[/yellow]")
            for model in blessed_models:
                console.print(f"  - {model}")
            return

        console.print(f"Downloading {hf_model_name}...")

        allow_patterns = None
        if gguf_config:
            allow_patterns = [
                gguf_config["gguf_file"],
                "*.json",
                ".gitattributes",
                "README.md",
                "LICENSE*",
            ]

        snapshot_download(
            repo_id=hf_model_name,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            force_download=force,
            token=HfFolder.get_token(),
        )

        console.print(f"[green]✓ {model_name} downloaded[/green]")

        # Download tokenizer for blessed GGUF models
        if gguf_config:
            _download_base_tokenizer(
                gguf_config["base_model"],
                gguf_config["tokenizer_file"],
                local_dir,
                force,
            )

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
