"""Add a new model to ROSE server."""

from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import get_client

console = Console()


def add_model(
    id: str = typer.Argument(..., help="Model ID (e.g., 'llama-3.2-1b')"),
    model_name: str = typer.Argument(..., help="HuggingFace model name (e.g., 'meta-llama/Llama-3.2-1B-Instruct')"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Default temperature"),
    top_p: float = typer.Option(0.9, "--top-p", "-p", help="Default top_p"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Timeout in seconds"),
    lora_modules: Optional[List[str]] = typer.Option(None, "--lora-modules", "-l", help="LoRA target modules"),
    owned_by: str = typer.Option("organization-owner", "--owned-by", "-o", help="Model owner"),
    quantization: Optional[str] = typer.Option(None, "--quantization", "-q", help="Quantization type (e.g., 'int8')"),
) -> None:
    """Add a new model configuration to ROSE server."""
    client = get_client()

    # Prepare request data
    data = {
        "id": id,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "owned_by": owned_by,
    }

    if timeout:
        data["timeout"] = timeout
    if lora_modules:
        data["lora_target_modules"] = lora_modules
    if quantization:
        data["quantization"] = quantization

    try:
        # Build auth headers from the client's API key
        headers = {}
        if client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"

        # Use the OpenAI client's underlying httpx client to make the request
        response = client._client.post(
            "/models",
            json=data,
            headers=headers,
        )
        response.raise_for_status()

        model = response.json()

        # Display success message
        console.print(f"[green]Successfully added model '{id}'[/green]")

        # Display model details
        table = Table(title="Model Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("ID", model["id"])
        table.add_row("Model Name", model_name)
        table.add_row("Display Name", name or id)
        table.add_row("Temperature", str(temperature))
        table.add_row("Top P", str(top_p))
        table.add_row("Timeout (s)", str(timeout) if timeout else "None")
        table.add_row("Owner", owned_by)
        table.add_row("Quantization", quantization or "None")
        table.add_row("Created", str(model["created"]))

        console.print(table)

    except Exception as e:
        if hasattr(e, "response") and hasattr(e.response, "json"):
            try:
                error_detail = e.response.json()
                console.print(f"[red]Error: {error_detail.get('detail', str(e))}[/red]")
            except Exception:
                console.print(f"[red]Error: {e}[/red]")
        else:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
