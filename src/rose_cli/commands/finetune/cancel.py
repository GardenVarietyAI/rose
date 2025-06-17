from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def cancel_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Cancel a fine-tuning job."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        job = client.fine_tuning.jobs.cancel(job_id)
        console.print(f"[red]Fine-tuning job {job_id} cancelled[/red]")
        console.print(f"Status: {job.status}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
