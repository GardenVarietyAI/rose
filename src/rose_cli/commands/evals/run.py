import time
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ...utils import get_client

console = Console()


def run_eval(
    eval_id: str,
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to evaluate"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Maximum number of samples to evaluate"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Temperature for generation"),
):
    """Run an evaluation on a model."""
    client = get_client()
    run_data = {
        "model_id": model,
        "metadata": {"temperature": temperature, "source": "cli"},
    }
    if max_samples:
        run_data["metadata"]["max_samples"] = max_samples
    try:
        response = client.post(f"/v1/evals/{eval_id}/runs", json=run_data)
        response.raise_for_status()
        run = response.json()
        run_id = run["id"]
        console.print(f"[green]Started evaluation run: {run_id}[/green]")
        console.print(f"Model: {model}")
        console.print(f"Temperature: {temperature}")
        if max_samples:
            console.print(f"Max samples: {max_samples}")
        console.print("\n[yellow]Monitoring progress...[/yellow]")
        # Monitor progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating", total=100)
            last_status = None
            while True:
                run_response = client.get(f"/v1/evals/{eval_id}/runs/{run_id}")
                run_response.raise_for_status()
                run_data = run_response.json()
                status = run_data.get("status", "unknown")
                if status != last_status:
                    last_status = status
                    progress.update(task, description=f"Status: {status}")
                # Try to estimate progress from the run data
                if "progress" in run_data:
                    progress.update(task, completed=run_data["progress"])
                elif "samples_completed" in run_data and "total_samples" in run_data:
                    completed = run_data["samples_completed"]
                    total = run_data["total_samples"]
                    if total > 0:
                        progress.update(task, completed=(completed / total) * 100)
                if status in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(2)
        # Show results
        if status == "completed":
            console.print("\n[green]Evaluation completed![/green]")
            # Get run details with scores
            run_response = client.get(f"/v1/evals/{eval_id}/runs/{run_id}")
            run_response.raise_for_status()
            run_data = run_response.json()
            if "result_summary" in run_data:
                summary = run_data["result_summary"]
                console.print(
                    Panel(
                        f"[bold]Results:[/bold]\n\n"
                        f"Total Samples: {summary.get('total_samples', 'N/A')}\n"
                        f"Success Rate: {summary.get('success_rate', 0):.2%}\n"
                        f"Average Score: {summary.get('average_score', 0):.3f}\n"
                        f"Duration: {summary.get('duration_seconds', 0):.1f}s",
                        title="Evaluation Summary",
                        expand=False,
                    )
                )
        else:
            console.print(f"\n[red]Evaluation {status}[/red]")
    except Exception as e:
        console.print(f"[red]Error running eval: {e}[/red]")
