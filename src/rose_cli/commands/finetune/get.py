import typer

from ...utils import console, get_client


def get_job(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the job status"),
    model_only: bool = typer.Option(False, "--model-only", help="Only output the fine-tuned model name"),
):
    """Get fine-tuning job status."""
    client = get_client()
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)

        if quiet:
            print(job.status)
        elif model_only:
            if job.fine_tuned_model:
                print(job.fine_tuned_model)
        else:
            console.print(f"[cyan]Job ID:[/cyan] {job.id}")
            console.print(f"[cyan]Status:[/cyan] {job.status}")
            console.print(f"[cyan]Model:[/cyan] {job.model}")
            console.print(f"[cyan]Fine-tuned Model:[/cyan] {job.fine_tuned_model or 'N/A'}")
            console.print(f"[cyan]Created:[/cyan] {job.created_at}")
            console.print(f"[cyan]Finished:[/cyan] {job.finished_at or 'N/A'}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
