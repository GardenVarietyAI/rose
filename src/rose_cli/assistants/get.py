import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def get_assistant(
    assistant_id: str = typer.Argument(..., help="Assistant ID"),
):
    """Get a specific assistant."""
    client = get_client()
    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
        console.print(f"[cyan]Assistant ID:[/cyan] {assistant.id}")
        console.print(f"[cyan]Name:[/cyan] {assistant.name}")
        console.print(f"[cyan]Model:[/cyan] {assistant.model}")
        console.print(f"[cyan]Temperature:[/cyan] {assistant.temperature}")
        if assistant.description:
            console.print(f"[cyan]Description:[/cyan] {assistant.description}")
        if assistant.instructions:
            console.print(f"[cyan]Instructions:[/cyan] {assistant.instructions}")
        if assistant.tools:
            tools = [t.type for t in assistant.tools]
            console.print(f"[cyan]Tools:[/cyan] {', '.join(tools)}")
        console.print(f"[cyan]Created:[/cyan] {assistant.created_at}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
