import typer
from rich import print

from rose_cli.actors.calculator.actor import CalculatorActor


def calculator(
    query: str = typer.Argument(..., help="Math question or expression"),
    model: str = typer.Option("Qwen--Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model to use"),
) -> None:
    """Run a calculator agent to solve math problems."""
    actor = CalculatorActor(model=model)

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
