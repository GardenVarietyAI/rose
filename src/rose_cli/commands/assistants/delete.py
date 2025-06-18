from typing import Optional

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def delete_assistant(
    assistant_id: str = typer.Argument(..., help="Assistant ID to delete"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Delete an assistant."""
    client = get_client()
    if not typer.confirm(f"Are you sure you want to delete assistant {assistant_id}?"):
        console.print("Cancelled.")
        return
    try:
        client.beta.assistants.delete(assistant_id)
        console.print(f"[red]Deleted assistant: {assistant_id}[/red]")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
