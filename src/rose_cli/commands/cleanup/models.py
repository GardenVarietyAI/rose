from datetime import datetime
from typing import Optional

import typer
from openai import OpenAI
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

console = Console()


def cleanup_models(
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
        console.print("\n[green]Model cleanup complete.[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
