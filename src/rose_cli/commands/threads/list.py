from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from ...utils import get_client

console = Console()


def list_threads(
    limit: int = typer.Option(20, help="Number of threads to list"),
    base_url: Optional[str] = typer.Option(None, help="Override base URL"),
):
    """List threads."""
    get_client(base_url)
    try:
        url = base_url or "http://localhost:8004"
        with httpx.Client() as client:
            response = client.get(f"{url}/v1/threads", params={"limit": limit})
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
                thread["created_at"],
                str(thread.get("message_count", 0)),
                metadata_str or "-",
            )

        console.print(table)
    except httpx.HTTPStatusError as e:
        console.print(f"HTTP Error: {e.response.status_code} - {e.response.text}", style="red")
    except Exception as e:
        console.print(f"Error: {e}", style="red")
