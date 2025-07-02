import typer

from rose_cli.responses.create import create_response
from rose_cli.responses.retrieve import retrieve_response

app = typer.Typer()

app.command(name="create")(create_response)
app.command(name="retrieve")(retrieve_response)
