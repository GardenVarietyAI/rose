from typing import List, Optional

import typer
from openai import OpenAI
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

console = Console()


def cleanup_files(
    purpose: Optional[str] = typer.Option(None, "--purpose", help="Only clean files with this purpose"),
    orphaned: bool = typer.Option(False, "--orphaned", help="Only clean orphaned files not used by any job"),
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean up uploaded files."""
    client = OpenAI(base_url=base_url, api_key="dummy")
    console.print("Scanning for files...")
    try:
        files_response = client.files.list()
        files_to_delete: List = []
        if purpose:
            files_to_delete = [f for f in files_response.data if f.purpose == purpose]
        elif orphaned:
            # Check which files are not referenced by any fine-tuning job
            jobs_response = client.fine_tuning.jobs.list(limit=100)
            used_file_ids = set()
            for job in jobs_response.data:
                if hasattr(job, "training_file") and job.training_file:
                    used_file_ids.add(job.training_file)
                if hasattr(job, "validation_file") and job.validation_file:
                    used_file_ids.add(job.validation_file)
            files_to_delete = [f for f in files_response.data if f.id not in used_file_ids]
        else:
            files_to_delete = list(files_response.data)
        if not files_to_delete:
            console.print("[green]No files found to clean up.[/green]")
            return
        table = Table(title="Files to Delete")
        table.add_column("File ID", style="cyan")
        table.add_column("Filename", style="white")
        table.add_column("Purpose", style="yellow")
        table.add_column("Size", style="green")
        for file in files_to_delete:
            size = f"{file.bytes:,}" if hasattr(file, "bytes") else "N/A"
            table.add_row(file.id, file.filename, file.purpose, size)
        console.print(table)
        if dry_run:
            console.print(f"\n[yellow]DRY RUN: Would delete {len(files_to_delete)} files[/yellow]")
            return
        if not Confirm.ask(f"\nDelete {len(files_to_delete)} files?"):
            console.print("Cancelled.")
            return
        for file in files_to_delete:
            try:
                client.files.delete(file.id)
                console.print(f"[green]✓[/green] Deleted {file.filename} ({file.id})")
            except Exception as e:
                console.print(f"[red]✗[/red] Error deleting {file.filename}: {e}")
        console.print("\n[green]File cleanup complete.[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
