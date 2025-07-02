import typer
from rich.table import Table

from rose_cli.utils import console, get_client


def list_jobs(
    table: bool = typer.Option(False, "--table", "-t", help="Show as table"),
) -> None:
    """List fine-tuning jobs."""
    client = get_client()
    try:
        jobs = client.fine_tuning.jobs.list()
        if table:
            table_obj = Table(title="Fine-tuning Jobs")
            table_obj.add_column("Job ID", style="cyan")
            table_obj.add_column("Status", style="green")
            table_obj.add_column("Base Model", style="yellow")
            table_obj.add_column("Fine-tuned Model", style="magenta")
            for job in jobs.data:
                status_style = "green" if job.status == "succeeded" else "yellow" if job.status == "running" else "red"
                table_obj.add_row(
                    job.id[:20] + "...",
                    f"[{status_style}]{job.status}[/{status_style}]",
                    job.model,
                    job.fine_tuned_model or "-",
                )
            console.print(table_obj)
        else:
            for job in jobs.data:
                console.print(f"{job.id}\t{job.status}\t{job.model}\t{job.fine_tuned_model or '-'}")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
