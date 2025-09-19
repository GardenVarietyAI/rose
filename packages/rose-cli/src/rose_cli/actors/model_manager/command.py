import typer
from rose_cli.actors.model_manager.actor import ModelManagerActor


def model_manager(
    query: str = typer.Argument(..., help="Query for the model manager"),
    model: str = typer.Option("Qwen--Qwen3-4B-GGUF", "--model", "-m", help="Model to use for the actor"),
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
