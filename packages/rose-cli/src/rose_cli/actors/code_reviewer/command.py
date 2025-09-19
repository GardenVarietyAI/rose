import typer
from rich import print
from rose_cli.actors.code_reviewer.actor import CodeReviewerActor


def code_reviewer(
    query: str = typer.Argument(..., help="Code review request (e.g., 'Review utils.py' or 'Refactor main.py')"),
    model: str = typer.Option("Qwen--Qwen3-4B-GGUF", "--model", "-m", help="Model to use"),
) -> None:
    """Run a code review and refactoring agent."""
    actor = CodeReviewerActor(model=model)

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
