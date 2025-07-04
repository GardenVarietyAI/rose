"""Seed default models into the database."""

from rich import print

from rose_cli.utils import get_client


def seed_models() -> None:
    """Seed default models into the database."""
    client = get_client()

    # Build auth headers from the client's API key
    headers = {}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    # Default models to seed
    default_models = [
        {
            "id": "phi-1.5",
            "name": "phi-1.5",
            "model_name": "microsoft/phi-1_5",
            "temperature": 0.7,
            "top_p": 0.95,
            "memory_gb": 2.5,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
            "owned_by": "microsoft",
        },
        {
            "id": "phi-2",
            "name": "phi-2",
            "model_name": "microsoft/phi-2",
            "temperature": 0.5,
            "top_p": 0.9,
            "memory_gb": 5.0,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
            "owned_by": "microsoft",
        },
        {
            "id": "qwen-coder",
            "name": "qwen-coder",
            "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "temperature": 0.2,
            "top_p": 0.9,
            "memory_gb": 3.0,
            "timeout": 90,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "owned_by": "alibaba",
        },
        {
            "id": "qwen2.5-0.5b",
            "name": "qwen2.5-0.5b",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "temperature": 0.3,
            "top_p": 0.9,
            "memory_gb": 1.5,
            "timeout": 60,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "owned_by": "alibaba",
        },
        {
            "id": "hermes-3-llama-3.2-3b",
            "name": "Hermes 3 Llama 3.2 3B",
            "model_name": "NousResearch/Hermes-3-Llama-3.2-3B",
            "temperature": 0.7,
            "top_p": 0.9,
            "memory_gb": 6.0,
            "timeout": 120,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "owned_by": "nous-research",
        },
    ]

    seeded_count = 0
    skipped_count = 0

    for model_data in default_models:
        model_id = model_data["id"]

        # Check if model already exists
        try:
            existing = client.models.retrieve(model_id)
            if existing:
                print(f"[yellow]Model '{model_id}' already exists, skipping[/yellow]")
                skipped_count += 1
                continue
        except Exception:
            # Model doesn't exist, proceed with creation
            pass

        try:
            response = client._client.post(
                "/models",
                json=model_data,
                headers=headers,
            )
            response.raise_for_status()
            print(f"[green]✓[/green] Seeded model: {model_id}")
            seeded_count += 1
        except Exception as e:
            print(f"[red]Failed to seed model '{model_id}': {e}[/red]")

    print("\n[bold]Seeding complete:[/bold]")
    print(f"  - Seeded: {seeded_count} models")
    print(f"  - Skipped: {skipped_count} models (already exist)")
