import typer

from rose_cli.threads.add_message import add_message
from rose_cli.threads.create import create_thread
from rose_cli.threads.delete import delete_thread
from rose_cli.threads.get import get_thread
from rose_cli.threads.list import list_threads
from rose_cli.threads.list_messages import list_messages

app = typer.Typer()

app.command(name="create")(create_thread)
app.command(name="list")(list_threads)
app.command(name="get")(get_thread)
app.command(name="add-message")(add_message)
app.command(name="list-messages")(list_messages)
app.command(name="delete")(delete_thread)
