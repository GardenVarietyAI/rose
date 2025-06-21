import time
from datetime import datetime

import typer

from rose_cli.utils import console, get_client


def show_events(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow events"),
):
    """Show fine-tuning job events."""
    client = get_client()
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        console.print(f"[yellow]Events for job {job_id} (status: {job.status})[/yellow]")

        seen_events = set()

        while True:
            events = client.fine_tuning.jobs.list_events(job_id)
            new_events = [e for e in events.data if e.id not in seen_events]

            for event in reversed(new_events):
                seen_events.add(event.id)
                timestamp = datetime.fromtimestamp(event.created_at).strftime("%Y-%m-%d %H:%M:%S")
                level_style = "green" if event.level == "info" else "red"
                console.print(
                    f"[dim]{timestamp}[/dim] [{level_style}]{event.level.upper()}[/{level_style}]: {event.message}"
                )

            if not follow:
                break

            # Check if job is complete
            job = client.fine_tuning.jobs.retrieve(job_id)
            if job.status in ["succeeded", "failed", "cancelled"]:
                console.print(f"[yellow]Job {job.status}. Exiting.[/yellow]")
                break

            time.sleep(5)

    except Exception as e:
        console.print(f"[red]error: {e}[/red]")
