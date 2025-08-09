import typer

from rose_cli.vectorstores.create import create_vectorstore
from rose_cli.vectorstores.list import list_vectorstores

app = typer.Typer()

app.command(name="list")(list_vectorstores)
app.command(name="create")(create_vectorstore)
