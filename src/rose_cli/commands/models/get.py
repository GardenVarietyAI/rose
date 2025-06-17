from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def get_model(
    model_id: str = typer.Argument(..., help="Model ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Get model details."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        model = client.models.retrieve(model_id)
        console.print(f"[cyan]ID:[/cyan] {model.id}")
        console.print(f"[cyan]Type:[/cyan] {model.object}")
        console.print(f"[cyan]Created:[/cyan] {model.created}")
        console.print(f"[cyan]Owned by:[/cyan] {model.owned_by}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
