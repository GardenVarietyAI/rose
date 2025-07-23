import time

import typer
from rich.table import Table

from rose_cli.utils import console, get_client

app = typer.Typer()


@app.command()
def list_vectorstores() -> None:
    """List vector stores."""
    client = get_client()
    try:
        vector_stores = client.vector_stores.list()
        table = Table(title="Vector Stores")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Dimensions", style="yellow")
        table.add_column("Created", style="blue")

        for vs in vector_stores.data:
            dimensions = str(getattr(vs, "dimensions", 0))
            created_at = getattr(vs, "created_at", 0)
            created_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at))
            table.add_row(vs.id, vs.name, dimensions, created_date)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
