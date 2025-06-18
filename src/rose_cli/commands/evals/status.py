import typer
from rich.console import Console

from ...utils import get_client

console = Console()


def eval_status(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID"),
    eval_id: str = typer.Option(..., "--eval-id", "-e", help="Evaluation ID"),
):
    """Check the status of an evaluation run."""
    client = get_client()
    try:
        run = client.evals.runs.retrieve(eval_id=eval_id, run_id=run_id)

        console.print(f"[cyan]Run ID:[/cyan] {run.id}")
        console.print(f"[cyan]Status:[/cyan] {run.status}")
        console.print(f"[cyan]Model:[/cyan] {run.model}")
        console.print(f"[cyan]Created:[/cyan] {run.created_at}")

        if run.error:
            console.print(f"[red]Error:[/red] {run.error}")

        # Show results if completed
        if run.status == "completed":
            if run.result_counts:
                console.print("\n[cyan]Result Counts:[/cyan]")
                console.print(f"  Total: {run.result_counts.total}")
                console.print(f"  Passed: {run.result_counts.passed}")
                console.print(f"  Failed: {run.result_counts.failed}")

            if run.per_testing_criteria_results:
                console.print("\n[cyan]Per Criteria Results:[/cyan]")
                for criteria in run.per_testing_criteria_results:
                    total = criteria.passed + criteria.failed
                    pass_rate = criteria.passed / total if total > 0 else 0
                    console.print(f"  {criteria.testing_criteria}: {criteria.passed}/{total} passed ({pass_rate:.2%})")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
