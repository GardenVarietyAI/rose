from typing import List

import typer

from rose_cli.utils import console, get_client


def create_vectorstore(
    name: str = typer.Argument(None, help="Name of the vector store"),
    file_ids: List[str] = typer.Option(..., "--files", "-f", help="Previously uploaded file IDs"),
) -> None:
    """Create a vector store."""
    client = get_client()
    try:
        vector_store = client.vector_stores.create(name=name, file_ids=file_ids)
        console.print(f"[green]Vector store {vector_store.name} created with ID {vector_store.id}[/green]")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
