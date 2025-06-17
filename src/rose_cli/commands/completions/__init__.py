import typer

from .completions import create_completion

app = typer.Typer()
app.callback(invoke_without_command=True)(create_completion)
