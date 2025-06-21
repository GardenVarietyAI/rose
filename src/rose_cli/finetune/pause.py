import typer

from rose_cli.utils import console, get_client


def pause_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
):
    """Pause a fine-tuning job."""
    client = get_client()
    try:
        # Use POST request directly since SDK doesn't have pause method
        client._client.post(f"/fine_tuning/jobs/{job_id}/pause")
        console.print(f"[yellow]Fine-tuning job {job_id} paused[/yellow]")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
