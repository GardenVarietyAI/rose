import typer

from rose_cli.files.delete import delete_file
from rose_cli.files.get import get_file
from rose_cli.files.list import list_files
from rose_cli.files.upload import upload_file

app = typer.Typer()

app.command(name="list")(list_files)
app.command(name="upload")(upload_file)
app.command(name="get")(get_file)
app.command(name="delete")(delete_file)
