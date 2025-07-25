import json
from typing import Optional

import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def add_message(
    thread_id: str = typer.Argument(..., help="Thread ID to add message to"),
    content: str = typer.Argument(..., help="Message content"),
    role: str = typer.Option("user", help="Message role (user/assistant)"),
    metadata_json: Optional[str] = typer.Option(None, help="Metadata as JSON string"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the message ID"),
) -> None:
    """Add a message to a thread."""
    client = get_client()
    metadata = {}
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            console.print("Invalid JSON for metadata", style="red")
            return

    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content,
            metadata=metadata,
        )

        if quiet:
            print(message.id)
        else:
            console.print(f"Created message: [green]{message.id}[/green]")
            console.print(f"Role: {message.role}")
            if message.content and len(message.content) > 0:
                content_block = message.content[0]
                if hasattr(content_block, "text") and hasattr(content_block.text, "value"):
                    console.print(f"Content: {content_block.text.value}")
            if message.metadata:
                console.print("Metadata:")
                for key, value in message.metadata.items():
                    console.print(f"   {key}: {value}")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
