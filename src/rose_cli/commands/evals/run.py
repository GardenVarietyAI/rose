from typing import Any

import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def run_eval(
    eval_id: str = typer.Argument(..., help="Evaluation ID to run"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to evaluate"),
):
    """Run an evaluation."""
    client = get_client()
    try:
        # Create run parameters
        params: Any = {
            "data_source": {
                "type": "eval",
                "eval_id": eval_id,
            }
        }

        # Create the run
        created_run = client.evals.runs.create(eval_id=eval_id, **params)

        console.print(f"[green]Started evaluation run: {created_run.id}[/green]")
        console.print(f"Status: {created_run.status}")
        console.print(f"Eval ID: {eval_id}")
        console.print(f"\nUse 'rose evals status {created_run.id}' to check progress")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
