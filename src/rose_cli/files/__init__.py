import typer

from .delete import delete_file
from .get import get_file
from .list import list_files
from .upload import upload_file

app = typer.Typer()

app.command(name="list")(list_files)
app.command(name="upload")(upload_file)
app.command(name="get")(get_file)
app.command(name="delete")(delete_file)
