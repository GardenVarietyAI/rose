from typing import Optional

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def create_thread(
    user_id: Optional[str] = typer.Option(None, help="User ID for the thread"),
    session_id: Optional[str] = typer.Option(None, help="Session ID"),
    conversation_type: Optional[str] = typer.Option(None, help="Type of conversation (chat, assistant, etc)"),
    source: Optional[str] = typer.Option(None, help="Source of the thread (api, web, cli)"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """Create a new thread."""
    client = get_client(base_url)
    metadata = {}
    if user_id:
        metadata["user_id"] = user_id
    if session_id:
        metadata["session_id"] = session_id
    if conversation_type:
        metadata["conversation_type"] = conversation_type
    if source:
        metadata["source"] = source
    try:
        thread = client.beta.threads.create(metadata=metadata)
        console.print(f"Created thread: [green]{thread.id}[/green]")
        if thread.metadata:
            console.print("Metadata:")
            for key, value in thread.metadata.items():
                console.print(f"   {key}: {value}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
