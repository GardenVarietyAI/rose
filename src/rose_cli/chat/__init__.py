import typer

from .chat import chat
from .interactive import interactive

app = typer.Typer()
app.callback(invoke_without_command=True)(chat)
app.command(name="interactive", help="Start an interactive chat session")(interactive)
