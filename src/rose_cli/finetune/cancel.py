import typer

from rose_cli.utils import console, get_client


def cancel_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
):
    """Cancel a fine-tuning job."""
    client = get_client()
    try:
        job = client.fine_tuning.jobs.cancel(job_id)
        console.print(f"[red]Fine-tuning job {job_id} cancelled[/red]")
        console.print(f"Status: {job.status}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
