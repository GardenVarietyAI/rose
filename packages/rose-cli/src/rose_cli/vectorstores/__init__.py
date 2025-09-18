import typer

from rose_cli.vectorstores.create import create_vectorstore
from rose_cli.vectorstores.delete import delete_vector_store
from rose_cli.vectorstores.delete_file import delete_vectorstore_file
from rose_cli.vectorstores.list import list_vectorstores
from rose_cli.vectorstores.list_files import list_vectorstore_files
from rose_cli.vectorstores.search import search_vectorstore
from rose_cli.vectorstores.update import update_vectorstore

app = typer.Typer()

app.command(name="list")(list_vectorstores)
app.command(name="create")(create_vectorstore)
app.command(name="update")(update_vectorstore)
app.command(name="delete")(delete_vector_store)
app.command(name="search")(search_vectorstore)
app.command(name="list-files")(list_vectorstore_files)
app.command(name="delete-file")(delete_vectorstore_file)
