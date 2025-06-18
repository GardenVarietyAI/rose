from typing import Optional

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def list_messages(
    thread_id: str = typer.Argument(..., help="Thread ID to list messages for"),
    limit: int = typer.Option(20, help="Number of messages to list"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """List messages in a thread."""
    client = get_client()
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id, limit=limit)
        for message in messages.data:
            console.print(f"\n[cyan]Message ID:[/cyan] {message.id}")
            console.print(f"[yellow]Role:[/yellow] {message.role}")
            console.print(f"[green]Created:[/green] {message.created_at}")
            if message.content:
                for content in message.content:
                    if hasattr(content, "text"):
                        console.print(f"[white]Content:[/white] {content.text.value}")
            if message.metadata:
                console.print("[blue]Metadata:[/blue]")
                for key, value in message.metadata.items():
                    console.print(f"   {key}: {value}")
            console.print("-" * 50)
    except Exception as e:
        console.print(f"Error: {e}", style="red")
