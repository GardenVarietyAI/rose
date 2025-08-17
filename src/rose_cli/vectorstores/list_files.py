import time

import typer
from rich.table import Table

from rose_cli.utils import console, get_client

app = typer.Typer()


@app.command()
def list_vectorstore_files(vector_store_id: str = typer.Argument(..., help="VectorStore ID to update")) -> None:
    """List vector stores."""
    client = get_client()
    try:
        vector_stores_files = client.vector_stores.files.list(vector_store_id=vector_store_id, order="desc")
        table = Table(title="Vector Store Files")
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Last Error", style="yellow")
        table.add_column("Created", style="blue")

        for vsf in vector_stores_files.data:
            status = getattr(vsf, "status", "unknown")
            last_error = str(getattr(vsf, "last_error", 0))
            created_at = getattr(vsf, "created_at", 0)
            created_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at))
            table.add_row(vsf.id, status, last_error, created_date)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
