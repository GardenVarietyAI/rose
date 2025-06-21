import typer

from .create import create_assistant
from .delete import delete_assistant
from .get import get_assistant
from .list import list_assistants
from .update import update_assistant

app = typer.Typer()

app.command(name="create")(create_assistant)
app.command(name="list")(list_assistants)
app.command(name="get")(get_assistant)
app.command(name="update")(update_assistant)
app.command(name="delete")(delete_assistant)
