import typer

from ...utils import console, get_client


def get_model(
    model_id: str = typer.Argument(..., help="Model ID"),
):
    """Get model details."""
    client = get_client()
    try:
        model = client.models.retrieve(model_id)
        console.print(f"[cyan]ID:[/cyan] {model.id}")
        console.print(f"[cyan]Type:[/cyan] {model.object}")
        console.print(f"[cyan]Created:[/cyan] {model.created}")
        console.print(f"[cyan]Owned by:[/cyan] {model.owned_by}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
