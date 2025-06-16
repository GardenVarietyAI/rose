from typing import Optional

import typer
from openai import OpenAI
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

console = Console()


def cleanup_jobs(
    status: Optional[str] = typer.Option(None, "--status", help="Only clean jobs with this status"),
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean up fine-tuning jobs by status."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    console.print("Scanning for fine-tuning jobs...")
    try:
        jobs_response = client.fine_tuning.jobs.list(limit=100)
        if status:
            jobs_to_delete = [job for job in jobs_response.data if job.status == status]
        else:
            # By default, clean up failed and cancelled jobs
            jobs_to_delete = [job for job in jobs_response.data if job.status in ["failed", "cancelled"]]
        if not jobs_to_delete:
            console.print("[green]No jobs found to clean up.[/green]")
            return
        table = Table(title="Jobs to Cancel/Delete")
        table.add_column("Job ID", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Model", style="white")
        table.add_column("Created", style="green")
        for job in jobs_to_delete:
            from datetime import datetime

            created = datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(job.id, job.status, job.model, created)
        console.print(table)
        if dry_run:
            console.print(f"\n[yellow]DRY RUN: Would process {len(jobs_to_delete)} jobs[/yellow]")
            return
        if not Confirm.ask(f"\nProcess {len(jobs_to_delete)} jobs?"):
            console.print("Cancelled.")
            return
        for job in jobs_to_delete:
            try:
                if job.status == "running":
                    # Cancel running jobs
                    client.fine_tuning.jobs.cancel(job.id)
                    console.print(f"[yellow]⚡[/yellow] Cancelled {job.id}")
                else:
                    # For failed/cancelled jobs, just mark as processed
                    console.print(f"[green]✓[/green] Processed {job.id} (status: {job.status})")
            except Exception as e:
                console.print(f"[red]✗[/red] Error processing {job.id}: {e}")
        console.print("\n[green]Job cleanup complete.[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
