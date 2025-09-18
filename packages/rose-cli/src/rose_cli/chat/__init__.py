import typer

from rose_cli.chat.chat import chat
from rose_cli.chat.interactive import interactive

app = typer.Typer()
app.callback(invoke_without_command=True)(chat)
app.command(name="interactive", help="Start an interactive chat session")(interactive)
