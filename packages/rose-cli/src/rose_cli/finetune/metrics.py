from typing import Any, Dict, Optional

import typer

from rose_cli.utils import console, get_client


def show_metrics(job_id: str = typer.Argument(..., help="Fine-tuning job ID")) -> None:
    """Show detailed training metrics for a fine-tuning job."""
    client = get_client()

    headers: Dict[str, str] = {}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    response = client._client.get(f"/fine_tuning/jobs/{job_id}", headers=headers)

    if response.status_code == 404:
        console.print(f"[red]Job {job_id} not found[/red]")
        return

    response.raise_for_status()
    job_data = response.json()

    _display_job_info(job_data)
    _display_training_metrics(job_data.get("training_metrics"))


def _display_job_info(job_data: Dict[str, Any]) -> None:
    """Display basic job information."""
    console.print(f"[bold cyan]Job:[/bold cyan] {job_data['id']}")
    console.print(f"[cyan]Status:[/cyan] {job_data['status']}")
    console.print(f"[cyan]Model:[/cyan] {job_data['model']}")
    console.print(f"[cyan]Fine-tuned Model:[/cyan] {job_data.get('fine_tuned_model') or 'None'}")


def _display_training_metrics(metrics: Optional[Dict[str, Any]]) -> None:
    """Display training metrics."""
    if not metrics:
        console.print("\n[yellow]No training metrics available.[/yellow]")
        return

    console.print("\n[bold green]Training Metrics:[/bold green]")

    summary = metrics.get("summary", {})
    console.print(f"[cyan]Final Loss:[/cyan] {summary.get('final_loss'):.4f}")
    console.print(f"[cyan]Steps:[/cyan] {summary.get('total_steps')}")
    console.print(f"[cyan]Epochs:[/cyan] {summary.get('epochs_completed')}")

    if summary.get("final_perplexity"):
        console.print(f"[cyan]Perplexity:[/cyan] {summary['final_perplexity']:.4f}")

    if summary.get("training_time_seconds"):
        console.print(f"[cyan]Training Time:[/cyan] {summary['training_time_seconds']:.1f}s")
