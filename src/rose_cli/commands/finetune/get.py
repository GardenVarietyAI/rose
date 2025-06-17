from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def get_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Get fine-tuning job status."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        console.print(f"[cyan]Job ID:[/cyan] {job.id}")
        console.print(f"[cyan]Status:[/cyan] {job.status}")
        console.print(f"[cyan]Model:[/cyan] {job.model}")
        console.print(f"[cyan]Fine-tuned Model:[/cyan] {job.fine_tuned_model or 'N/A'}")
        console.print(f"[cyan]Created:[/cyan] {job.created_at}")
        console.print(f"[cyan]Finished:[/cyan] {job.finished_at or 'N/A'}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
