import time

from rich.table import Table

from rose_cli.utils import console, get_client


def list_models() -> None:
    """List available models."""
    client = get_client()
    try:
        response = client.models.list()
        table = Table(title="Available Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Created", style="green")
        for model in response.data:
            created_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(model.created))
            table.add_row(model.id, created_date)
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
