"""Seed default models into the database."""

from rich import print

from rose_cli.models.download import download_model
from rose_cli.utils import get_client


def seed_models() -> None:
    """Seed default models into the database."""
    client = get_client()

    # Build auth headers from the client's API key
    headers = {}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    # Models to download from HuggingFace
    models_to_download = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-0.6B-GGUF",
        "janhq/Jan-v1-4B-GGUF",
    ]

    # Default models to seed
    default_models = [
        {
            "model_name": "microsoft/phi-1_5",
            "temperature": 0.7,
            "top_p": 0.95,
            "memory_gb": 2.5,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
        },
        {
            "model_name": "microsoft/phi-2",
            "temperature": 0.5,
            "top_p": 0.9,
            "memory_gb": 5.0,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
        },
        {
            "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "temperature": 0.2,
            "top_p": 0.9,
            "memory_gb": 3.0,
            "timeout": 90,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "temperature": 0.3,
            "top_p": 0.9,
            "memory_gb": 1.5,
            "timeout": 60,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "temperature": 0.3,
            "top_p": 0.9,
            "memory_gb": 3.0,
            "timeout": 90,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        {
            "model_name": "NousResearch/Hermes-3-Llama-3.2-3B",
            "temperature": 0.7,
            "top_p": 0.9,
            "memory_gb": 6.0,
            "timeout": 120,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    ]

    # Download models from HuggingFace first
    print("[bold]Downloading models from HuggingFace collection...[/bold]")
    downloaded_count = 0
    for model_name in models_to_download:
        try:
            print(f"Downloading {model_name}...")
            download_model(model_name, force=False)
            downloaded_count += 1
        except SystemExit:
            # download_model raises SystemExit on error, but we want to continue
            print(f"[red]Failed to download {model_name}[/red]")
        except Exception as e:
            if "already downloaded" in str(e):
                print(f"[dim]{model_name} already exists[/dim]")
            else:
                print(f"[red]Failed to download {model_name}: {e}[/red]")

    print(f"[green]✓ Downloaded {downloaded_count} new models[/green]\n")

    # Seed models into database
    print("[bold]Seeding models into database...[/bold]")
    seeded_count = 0

    for model_data in default_models:
        model_name = model_data["model_name"]

        try:
            response = client._client.post(
                "/models",
                json=model_data,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            print(f"[green]✓[/green] Seeded model: {model_name} (ID: {result['id']})")
            seeded_count += 1
        except Exception as e:
            if "already exists" in str(e):
                print(f"[dim]Model {model_name} already exists in database[/dim]")
            else:
                print(f"[red]Failed to seed model '{model_name}': {e}[/red]")

    print(f"\n[bold]Seeded {seeded_count} new models into database[/bold]")
