import typer

from ...utils import console, get_client


def resume_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
):
    """Resume a fine-tuning job."""
    client = get_client()
    try:
        # Use POST request directly since SDK doesn't have resume method
        client._client.post(f"/fine_tuning/jobs/{job_id}/resume")
        console.print(f"[green]Fine-tuning job {job_id} resumed[/green]")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
