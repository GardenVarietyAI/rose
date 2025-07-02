import typer

from rose_cli.actors.calculator_actor import calculator

app = typer.Typer()

app.command(name="calculator-actor")(calculator)
