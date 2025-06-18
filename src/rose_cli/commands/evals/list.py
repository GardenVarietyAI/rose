import typer
from rich.console import Console
from rich.table import Table

from ...utils import get_client

console = Console()


def list_evals(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of evals to list"),
    order: str = typer.Option("desc", "--order", "-o", help="Sort order (asc/desc)"),
):
    """List evaluations."""
    client = get_client()
    try:
        evals = client.evals.list(
            limit=limit,
            order=order,  # type: ignore
        )

        if not evals.data:
            console.print("No evaluations found.")
            return

        table = Table(title="Evaluations")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Created", style="blue")

        for eval_item in evals.data:
            table.add_row(
                eval_item.id,
                eval_item.name or "-",
                str(eval_item.created_at),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
