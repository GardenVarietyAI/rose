from typing import Optional

import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def create_thread(
    user_id: Optional[str] = typer.Option(None, help="User ID for the thread"),
    session_id: Optional[str] = typer.Option(None, help="Session ID"),
    conversation_type: Optional[str] = typer.Option(None, help="Type of conversation (chat, assistant, etc)"),
    source: Optional[str] = typer.Option(None, help="Source of the thread (api, web, cli)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the thread ID"),
) -> None:
    """Create a new thread."""
    client = get_client()
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

        if quiet:
            print(thread.id)
        else:
            console.print(f"Created thread: [green]{thread.id}[/green]")
            if thread.metadata:
                console.print("Metadata:")
                for key, value in thread.metadata.items():
                    console.print(f"   {key}: {value}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
