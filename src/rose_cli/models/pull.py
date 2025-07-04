import typer

from rose_cli.utils import console, get_client


def pull_model(
    model_name: str = typer.Argument(..., help="Model name to pull"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download even if exists"),
) -> None:
    """Download a model to local storage."""
    console.print(f"[yellow]Downloading model: {model_name}...[/yellow]")

    client = get_client()

    # Build auth headers
    headers = {}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    try:
        # Call the download endpoint
        response = client._client.post(
            "/models/download",
            json={"model": model_name, "force": force},
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()

        if result["status"] == "exists":
            console.print(f"[yellow]{result['message']}[/yellow]")
        else:
            console.print(f"[green]{result['message']}[/green]")

        console.print(f"[dim]Path: {result['path']}[/dim]")

    except Exception as e:
        if hasattr(e, "response") and hasattr(e.response, "json"):
            try:
                error_data = e.response.json()
                console.print(f"[red]Error: {error_data.get('detail', str(e))}[/red]")
            except Exception:
                console.print(f"[red]Error: {e}[/red]")
        else:
            console.print(f"[red]Error: {e}[/red]")
