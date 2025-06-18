import typer

from .calcular_actor import calculator

app = typer.Typer()

app.command(name="calculator-actor")(calculator)
