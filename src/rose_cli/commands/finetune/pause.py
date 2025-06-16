from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def pause_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Pause a fine-tuning job."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        # Use POST request directly since SDK doesn't have pause method
        client._client.post(f"/fine_tuning/jobs/{job_id}/pause")
        console.print(f"[yellow]Fine-tuning job {job_id} paused[/yellow]")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
