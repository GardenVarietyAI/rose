import httpx
import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import BASE_URL

console = Console()


def list_threads(
    limit: int = typer.Option(20, help="Number of threads to list"),
):
    """List threads."""
    try:
        with httpx.Client() as client:
            response = client.get(f"{BASE_URL}/threads", params={"limit": limit})
            response.raise_for_status()
            data = response.json()

        threads = data.get("data", [])
        if not threads:
            console.print("No threads found.")
            return

        table = Table(title="Threads")
        table.add_column("Thread ID", style="cyan")
        table.add_column("Created At", style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Metadata", style="blue")

        for thread in threads:
            metadata_str = ""
            if thread.get("metadata"):
                metadata_items = [f"{k}={v}" for k, v in thread["metadata"].items()]
                metadata_str = ", ".join(metadata_items)

            table.add_row(
                thread["id"],
                str(thread["created_at"]),
                str(thread.get("message_count", 0)),
                metadata_str or "-",
            )

        console.print(table)
    except Exception as e:
        console.print(f"Error: {e}", style="red")
