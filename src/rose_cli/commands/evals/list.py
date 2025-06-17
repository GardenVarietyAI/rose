from rich import box
from rich.console import Console
from rich.table import Table

from ...utils import get_client

console = Console()


def list_evals():
    """List all evaluations."""
    client = get_client()
    try:
        response = client.get("/v1/evals")
        response.raise_for_status()
        evals = response.json()
        if not evals.get("data"):
            console.print("[yellow]No evaluations found.[/yellow]")
            return
        table = Table(title="Evaluations", box=box.ROUNDED)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Created", style="blue")
        for eval_item in evals["data"]:
            created_at = eval_item.get("created_at", "N/A")
            if isinstance(created_at, (int, float)):
                from datetime import datetime

                created_at = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                eval_item.get("id", "N/A"),
                eval_item.get("name", "N/A"),
                eval_item.get("data_source", {}).get("type", "N/A"),
                created_at,
            )
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing evals: {e}[/red]")
