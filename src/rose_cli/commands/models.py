"""Models command."""
import time
from typing import Optional
import typer
from rich.table import Table
from ..utils import console, get_client, get_endpoint_url
app = typer.Typer()
@app.command()

def list(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    table: bool = typer.Option(False, "--table", "-t", help="Show as table"),
):
    """List available models."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        models = client.models.list()
        if table:
            table_obj = Table(title="Available Models")
            table_obj.add_column("Model ID", style="cyan")
            table_obj.add_column("Owner", style="dim")
            table_obj.add_column("Created", style="dim")
            for model in models.data:
                created = time.strftime("%Y-%m-%d", time.localtime(model.created))
                owner = getattr(model, "owned_by", "system")
                table_obj.add_row(model.id, owner, created)
            console.print(table_obj)
        else:
            for model in models.data:
                console.print(model.id)
    except Exception as e:
        console.print(f"[red]error: {e}[/red]", file=typer.get_text_stream("stderr"))
@app.command()

def download(
    model_name: str = typer.Argument(..., help="Model name to download (e.g., tinyllama, qwen2.5-0.5b)"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Pre-download a model to avoid blocking during inference."""
    endpoint_url = get_endpoint_url(url, local)
    import httpx
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{endpoint_url}/v1/models/{model_name}/download"
            )
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                if status == "ready":
                    console.print(f"[green]✓[/green] Model {model_name} is already loaded and ready")
                elif status == "downloading":
                    console.print(f"[yellow]⏳[/yellow] Model {model_name} download started in background")
                    console.print("[dim]Use 'rose chat' to test when ready[/dim]")
                else:
                    console.print(f"[green]✓[/green] {data.get('message', 'Download initiated')}")
            elif response.status_code == 404:
                console.print(f"[red]error: Model '{model_name}' not found[/red]")
                console.print("[dim]Use 'rose models list' to see available models[/dim]")
            else:
                error_data = response.json()
                console.print(f"[red]error: {error_data.get('detail', 'Unknown error')}[/red]")
    except httpx.ConnectError:
        console.print("[red]error: Cannot connect to rose-server. Is it running?[/red]")
    except Exception as e:
        console.print(f"[red]error: {e}[/red]")