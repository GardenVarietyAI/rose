import typer
from rich import print

from rose_cli.actors.file_reader.actor import FileReaderActor


def file_reader(
    query: str = typer.Argument(..., help="File operation request"),
    model: str = typer.Option("qwen2.5-0.5b", "--model", "-m", help="Model to use"),
) -> None:
    """Run a file reader agent to list directories and read files."""
    actor = FileReaderActor(model=model)

    try:
        result = actor.run(query)
        if result.get("success", True):
            print(result["response"])
        else:
            print(f"[red]{result['response']}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
