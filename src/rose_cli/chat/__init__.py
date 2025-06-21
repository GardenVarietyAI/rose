import typer

from .chat import chat

app = typer.Typer()
app.callback(invoke_without_command=True)(chat)
