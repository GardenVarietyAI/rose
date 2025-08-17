import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def delete_vectorstore_file(
    vector_store_id: str = typer.Argument(..., help="VectorStore ID associated with the file to delete."),
    vector_store_file_id: str = typer.Argument(..., help="VectorStoreFile ID to delete"),
) -> None:
    """Delete a thread and all its messages."""
    client = get_client()
    if not typer.confirm(f"Are you sure you want to delete vector store file {vector_store_file_id}?"):
        console.print("Cancelled.")
        return
    try:
        client.vector_stores.files.delete(file_id=vector_store_file_id, vector_store_id=vector_store_id)
        console.print(f"[red]Deleted file: {vector_store_file_id} from vector store {vector_store_id}[/red]")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
