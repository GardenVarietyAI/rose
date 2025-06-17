import time
from typing import Optional

import typer
from rich.table import Table

from ...utils import console, get_client, get_endpoint_url


def list_models(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """List available models."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        response = client.models.list()
        table = Table(title="Available Models")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Created", style="green")
        for model in response.data:
            created_date = time.strftime("%Y-%m-%d %H:%M", time.localtime(model.created))
            table.add_row(model.id, model.object, created_date)
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
