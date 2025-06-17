import typer

from .add_message import add_message
from .create import create_thread
from .delete import delete_thread
from .get import get_thread
from .list import list_threads
from .list_messages import list_messages

app = typer.Typer()

app.command(name="create")(create_thread)
app.command(name="list")(list_threads)
app.command(name="get")(get_thread)
app.command(name="add-message")(add_message)
app.command(name="list-messages")(list_messages)
app.command(name="delete")(delete_thread)
