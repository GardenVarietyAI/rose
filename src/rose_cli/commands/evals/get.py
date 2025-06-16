import json

from rich.console import Console
from rich.panel import Panel

from ...utils import get_client

console = Console()


def get_eval(eval_id: str):
    """Get details of a specific evaluation."""
    client = get_client()
    try:
        response = client.get(f"/v1/evals/{eval_id}")
        response.raise_for_status()
        eval_data = response.json()
        console.print(
            Panel(
                f"[bold cyan]Evaluation: {eval_data['name']}[/bold cyan]\n\n"
                f"[yellow]ID:[/yellow] {eval_data['id']}\n"
                f"[yellow]Created:[/yellow] {eval_data['created_at']}\n"
                f"[yellow]Data Source:[/yellow] {eval_data['data_source']['type']}\n"
                f"[yellow]Metadata:[/yellow]\n{json.dumps(eval_data.get('metadata', {}), indent=2)}",
                title="Evaluation Details",
                expand=False,
            )
        )
        # Get runs for this eval
        runs_response = client.get(f"/v1/evals/{eval_id}/runs")
        runs_response.raise_for_status()
        runs_data = runs_response.json()
        if runs_data.get("data"):
            console.print("\n[bold]Runs:[/bold]")
            for run in runs_data["data"]:
                status = run.get("status", "unknown")
                status_color = "green" if status == "completed" else "yellow" if status == "running" else "red"
                console.print(
                    f"  • Run {run['id']}: [{status_color}]{status}[/{status_color}] "
                    f"(Model: {run.get('model_id', 'N/A')})"
                )
        else:
            console.print("\n[dim]No runs found for this evaluation.[/dim]")
    except Exception as e:
        console.print(f"[red]Error getting eval: {e}[/red]")
