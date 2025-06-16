"""Cleanup commands for managing models, files, and jobs."""
from typing import List, Optional
import typer
from openai import OpenAI
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
console = Console()
app = typer.Typer()
@app.command()

def models(
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean up fine-tuned models."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    console.print("Scanning for fine-tuned models...")
    try:
        models_response = client.models.list()
        fine_tuned_models = [
            model for model in models_response.data if model.id.startswith("qwen") and "ft-" in model.id
        ]
        if not fine_tuned_models:
            console.print("[green]No fine-tuned models found to clean up.[/green]")
            return
        table = Table(title="Fine-Tuned Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Created", style="white")
        table.add_column("Owner", style="yellow")
        for model in fine_tuned_models:
            from datetime import datetime
            created = datetime.fromtimestamp(model.created).strftime("%Y-%m-%d %H:%M")
            table.add_row(model.id, created, model.owned_by)
        console.print(table)
        if dry_run:
            console.print(f"\n[yellow]DRY RUN: Would delete {len(fine_tuned_models)} models[/yellow]")
            return
        if not Confirm.ask(f"\nDelete {len(fine_tuned_models)} fine-tuned models?"):
            console.print("Cancelled.")
            return
        for model in fine_tuned_models:
            try:
                response = client.models.delete(model.id)
                if response.deleted:
                    console.print(f"[green]✓[/green] Deleted {model.id}")
                else:
                    console.print(f"[red]✗[/red] Failed to delete {model.id}")
            except Exception as e:
                console.print(f"[red]✗[/red] Error deleting {model.id}: {e}")
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
@app.command()

def files(
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean up uploaded files."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    console.print("Scanning for uploaded files...")
    try:
        files_response = client.files.list()
        uploaded_files = files_response.data
        if not uploaded_files:
            console.print("[green]No uploaded files found to clean up.[/green]")
            return
        table = Table(title="Uploaded Files")
        table.add_column("File ID", style="cyan")
        table.add_column("Filename", style="white")
        table.add_column("Purpose", style="yellow")
        table.add_column("Size", style="green")
        for file in uploaded_files:
            size = f"{file.bytes / 1024:.1f} KB" if file.bytes else "Unknown"
            table.add_row(file.id, file.filename, file.purpose, size)
        console.print(table)
        if dry_run:
            console.print(f"\n[yellow]DRY RUN: Would delete {len(uploaded_files)} files[/yellow]")
            return
        if not Confirm.ask(f"\nDelete {len(uploaded_files)} uploaded files?"):
            console.print("Cancelled.")
            return
        for file in uploaded_files:
            try:
                response = client.files.delete(file.id)
                if response.deleted:
                    console.print(f"[green]✓[/green] Deleted {file.filename}")
                else:
                    console.print(f"[red]✗[/red] Failed to delete {file.filename}")
            except Exception as e:
                console.print(f"[red]✗[/red] Error deleting {file.filename}: {e}")
    except Exception as e:
        console.print(f"[red]Error listing files: {e}[/red]")
@app.command()

def jobs(
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
    status: List[str] = typer.Option(["failed", "cancelled"], "--status", help="Job statuses to clean up"),
):
    """Clean up fine-tuning jobs by status."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    console.print(f"Scanning for fine-tuning jobs with status: {', '.join(status)}...")
    try:
        jobs_response = client.fine_tuning.jobs.list()
        jobs_to_clean = [job for job in jobs_response.data if job.status in status]
        if not jobs_to_clean:
            console.print(f"[green]No jobs found with status: {', '.join(status)}[/green]")
            return
        table = Table(title="Fine-Tuning Jobs to Clean")
        table.add_column("Job ID", style="cyan")
        table.add_column("Model", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="green")
        for job in jobs_to_clean:
            from datetime import datetime
            created = datetime.fromtimestamp(job.created_at).strftime("%Y-%m-%d %H:%M")
            table.add_row(job.id, job.model, job.status, created)
        console.print(table)
        if dry_run:
            console.print(f"\n[yellow]DRY RUN: Would delete {len(jobs_to_clean)} jobs[/yellow]")
            return
        if not Confirm.ask(f"\nDelete {len(jobs_to_clean)} fine-tuning jobs?"):
            console.print("Cancelled.")
            return
        import httpx
        for job in jobs_to_clean:
            try:
                response = httpx.delete(f"{base_url}/fine_tuning/jobs/{job.id}")
                if response.status_code == 200:
                    console.print(f"[green]✓[/green] Deleted job {job.id}")
                else:
                    console.print(f"[red]✗[/red] Failed to delete job {job.id}: {response.text}")
            except Exception as e:
                console.print(f"[red]✗[/red] Error deleting job {job.id}: {e}")
    except Exception as e:
        console.print(f"[red]Error listing jobs: {e}[/red]")
@app.command()

def all(
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean up everything: failed jobs, orphaned files, and old models."""
    console.print("[bold]Full cleanup operation[/bold]")
    console.print("\nCleaning up failed/cancelled jobs...")
    jobs(base_url=base_url, dry_run=dry_run, status=["failed", "cancelled"])
    console.print("\nCleaning up orphaned files...")
    files(base_url=base_url, dry_run=dry_run)
    console.print("\nCleaning up old fine-tuned models...")
    if not dry_run and Confirm.ask("Also clean up fine-tuned models?"):
        models(base_url=base_url, dry_run=dry_run)
    console.print("\n[green]Cleanup complete![/green]")
if __name__ == "__main__":
    app()