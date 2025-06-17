from typing import Optional

import typer

from ...utils import console, get_client, get_endpoint_url


def pull_model(
    model_name: str = typer.Argument(..., help="Model name to pull"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Pre-download a model to avoid blocking during inference."""
    endpoint_url = get_endpoint_url(url, local)
    console.print(f"[yellow]Pulling model: {model_name}...[/yellow]")

    # Trigger model download by making a minimal inference request
    client = get_client(endpoint_url)
    try:
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
        )
        console.print(f"[green]Model {model_name} ready![/green]")
    except Exception as e:
        console.print(f"[red]Error pulling model: {e}[/red]")
