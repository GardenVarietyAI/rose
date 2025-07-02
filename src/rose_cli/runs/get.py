import typer
from rich.console import Console

from rose_cli.utils import get_client

console = Console()


def get_run(
    thread_id: str = typer.Argument(..., help="Thread ID"),
    run_id: str = typer.Argument(..., help="Run ID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the run status"),
) -> None:
    """Get a specific run."""
    client = get_client()
    try:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

        if quiet:
            print(run.status)
        else:
            console.print(f"[cyan]Run ID:[/cyan] {run.id}")
            console.print(f"[cyan]Status:[/cyan] {run.status}")
            console.print(f"[cyan]Assistant:[/cyan] {run.assistant_id}")
            console.print(f"[cyan]Model:[/cyan] {run.model}")
            console.print(f"[cyan]Created:[/cyan] {run.created_at}")

            if run.completed_at:
                console.print(f"[cyan]Completed:[/cyan] {run.completed_at}")

            if run.last_error:
                console.print(f"[red]Error:[/red] {run.last_error}")

            if run.status == "requires_action" and run.required_action:
                console.print("[yellow]Required action:[/yellow]")
                if hasattr(run.required_action, "submit_tool_outputs"):
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    for call in tool_calls:
                        console.print(f"  - {call.function.name}({call.function.arguments})")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
