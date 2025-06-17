import typer

from .compare import compare

app = typer.Typer()
app.callback(invoke_without_command=True)(compare)
