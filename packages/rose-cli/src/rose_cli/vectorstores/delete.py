import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def delete_vector_store(
    vector_store_id: str = typer.Argument(..., help="VectorStore ID to delete"),
) -> None:
    """Delete a vector store."""
    client = get_client()
    if not typer.confirm(f"Are you sure you want to delete vector store {vector_store_id}?"):
        console.print("Cancelled.")
        return
    try:
        client.vector_stores.delete(vector_store_id)
        console.print(f"[red]Deleted vctor store : {vector_store_id}[/red]")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
