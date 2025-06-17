import time
from datetime import datetime
from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def show_events(
    job_id: str = typer.Argument(..., help="Fine-tuning job ID"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow events"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Show fine-tuning job events."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
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
