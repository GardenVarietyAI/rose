from typing import Any, Optional

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def update_assistant(
    assistant_id: str = typer.Argument(..., help="Assistant ID to update"),
    name: Optional[str] = typer.Option(None, help="New name"),
    model: Optional[str] = typer.Option(None, help="New model"),
    instructions: Optional[str] = typer.Option(None, help="New instructions"),
    description: Optional[str] = typer.Option(None, help="New description"),
    temperature: Optional[float] = typer.Option(None, help="New temperature (0.0-2.0)"),
):
    """Update an assistant."""
    client = get_client()
    update_data: dict[str, Any] = {}
    if name is not None:
        update_data["name"] = name
    if model is not None:
        update_data["model"] = model
    if instructions is not None:
        update_data["instructions"] = instructions
    if description is not None:
        update_data["description"] = description
    if temperature is not None:
        update_data["temperature"] = temperature

    if not update_data:
        console.print("[yellow]No updates specified.[/yellow]")
        return

    try:
        assistant = client.beta.assistants.update(assistant_id, **update_data)
        console.print(f"[green]Updated assistant: {assistant.id}[/green]")
        if name:
            console.print(f"Name: {assistant.name}")
        if model:
            console.print(f"Model: {assistant.model}")
        if temperature is not None:
            console.print(f"Temperature: {assistant.temperature}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
