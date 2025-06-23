import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def run_eval(
    eval_id: str = typer.Argument(..., help="Evaluation ID to run"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to evaluate"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the run ID"),
) -> None:
    """Run an evaluation."""
    client = get_client()
    try:
        # Create the run with proper data_source
        created_run = client.evals.runs.create(
            eval_id=eval_id,
            name=f"Eval run for {model}",
            data_source={
                "type": "model",
                "model": model,
            },
        )

        if quiet:
            print(created_run.id)
        else:
            console.print(f"[green]Started evaluation run: {created_run.id}[/green]")
            console.print(f"Status: {created_run.status}")
            console.print(f"Model: {model}")
            console.print(f"Eval ID: {eval_id}")
            console.print(f"\nUse 'rose eval status {created_run.id}' to check progress")

    except Exception as e:
        if not quiet:
            console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)
