import typer

from .actor import ModelManagerActor


def model_manager(
    query: str = typer.Argument(..., help="Query for the model manager"),
    model: str = typer.Option("Qwen--Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model to use for the actor"),
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
