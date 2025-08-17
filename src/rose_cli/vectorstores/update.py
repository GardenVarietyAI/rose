from typing import Optional

import typer

from rose_cli.utils import console, get_client


def update_vectorstore(
    vector_store_id: str = typer.Argument(..., help="VectorStore ID to update"),
    name: Optional[str] = typer.Option(None, help="New name"),
) -> None:
    """Update a vector store."""
    client = get_client()
    try:
        vector_store = client.vector_stores.update(vector_store_id=vector_store_id, name=name)
        console.print(f"[green]Vector store {vector_store.name} updated with ID {vector_store.id}[/green]")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
