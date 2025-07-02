import typer
from rich.console import Console
from rich.table import Table

from rose_cli.utils import get_client

console = Console()


def list_runs(
    thread_id: str = typer.Argument(..., help="Thread ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of runs to list"),
) -> None:
    """List runs in a thread."""
    client = get_client()
    try:
        runs = client.beta.threads.runs.list(thread_id=thread_id, limit=limit)

        if not runs.data:
            console.print("[yellow]No runs found[/yellow]")
            return

        table = Table(title=f"Runs for thread {thread_id}")
        table.add_column("Run ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Assistant ID", style="yellow")
        table.add_column("Created", style="blue")

        for run in runs.data:
            status_style = "green" if run.status == "completed" else "red" if run.status == "failed" else "yellow"
            table.add_row(
                run.id,
                f"[{status_style}]{run.status}[/{status_style}]",
                run.assistant_id,
                str(run.created_at),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
