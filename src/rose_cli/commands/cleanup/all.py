from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm

from .files import cleanup_files
from .jobs import cleanup_jobs
from .models import cleanup_models

console = Console()


def cleanup_all(
    base_url: Optional[str] = typer.Option("http://localhost:8004/v1", "--base-url", help="Base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
):
    """Clean up everything: failed jobs, orphaned files, and old models."""
    console.print("[bold yellow]Full cleanup: jobs, files, and models[/bold yellow]\n")

    if not dry_run:
        if not Confirm.ask("This will clean up failed jobs, orphaned files, and old models. Continue?"):
            console.print("Cancelled.")
            return

    # Clean up failed/cancelled jobs first
    console.print("\n[bold cyan]Step 1: Cleaning up failed/cancelled jobs...[/bold cyan]")
    cleanup_jobs(status=None, base_url=base_url, dry_run=dry_run)

    # Clean up orphaned files
    console.print("\n[bold cyan]Step 2: Cleaning up orphaned files...[/bold cyan]")
    cleanup_files(orphaned=True, base_url=base_url, dry_run=dry_run)

    # Clean up old fine-tuned models
    console.print("\n[bold cyan]Step 3: Cleaning up fine-tuned models...[/bold cyan]")
    cleanup_models(base_url=base_url, dry_run=dry_run)

    console.print("\n[bold green]Full cleanup complete![/bold green]")
