import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def get_thread(
    thread_id: str = typer.Argument(..., help="Thread ID to get details for"),
):
    """Get a specific thread."""
    client = get_client()
    try:
        thread = client.beta.threads.retrieve(thread_id)
        console.print(f"Thread ID: [cyan]{thread.id}[/cyan]")
        console.print(f"Created At: {thread.created_at}")
        if thread.metadata:
            console.print("Metadata:")
            for key, value in thread.metadata.items():
                console.print(f"   {key}: {value}")
        else:
            console.print("Metadata: None")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
