import typer

from .actor import ModelManagerActor


def model_manager(
    query: str = typer.Argument(..., help="Query for the model manager"),
    model: str = typer.Option("qwen2.5-0.5b", "--model", "-m", help="Model to use for the actor"),
) -> None:
    """Manage models through the ROSE API."""
    try:
        actor = ModelManagerActor(model=model)
        result = actor.run(query)

        if result["success"]:
            typer.echo(result["response"])
        else:
            typer.echo(f"Error: {result['response']}")
    except Exception as e:
        typer.echo(f"Failed to run model manager: {str(e)}")
        raise typer.Exit(1)
