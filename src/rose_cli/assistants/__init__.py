import typer

from rose_cli.assistants.create import create_assistant
from rose_cli.assistants.delete import delete_assistant
from rose_cli.assistants.get import get_assistant
from rose_cli.assistants.list import list_assistants
from rose_cli.assistants.update import update_assistant

app = typer.Typer()

app.command(name="create")(create_assistant)
app.command(name="list")(list_assistants)
app.command(name="get")(get_assistant)
app.command(name="update")(update_assistant)
app.command(name="delete")(delete_assistant)
