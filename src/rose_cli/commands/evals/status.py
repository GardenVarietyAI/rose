import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def eval_status(
    run_id: str = typer.Argument(..., help="Run ID"),
):
    """Check the status of an evaluation run."""
    client = get_client()
    try:
        # Get run status - we'll need to use the low-level API
        # since we don't have the eval_id
        response = client._client.get(f"/v1/evals/runs/{run_id}")
        run = response.json()

        console.print(f"[cyan]Run ID:[/cyan] {run['id']}")
        console.print(f"[cyan]Status:[/cyan] {run['status']}")
        console.print(f"[cyan]Created:[/cyan] {run.get('created_at', '-')}")

        if run.get("completed_at"):
            console.print(f"[cyan]Completed:[/cyan] {run['completed_at']}")

        if run.get("error"):
            console.print(f"[red]Error:[/red] {run['error']}")

        # Show progress if available
        if run.get("progress_percent") is not None:
            console.print(f"[cyan]Progress:[/cyan] {run['progress_percent']}%")

        # Show results if completed
        if run["status"] == "completed" and run.get("results"):
            console.print("\n[cyan]Results:[/cyan]")
            for key, value in run["results"].items():
                console.print(f"  {key}: {value}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
