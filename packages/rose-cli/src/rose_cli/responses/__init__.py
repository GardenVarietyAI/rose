import typer

from rose_cli.responses.create import create_response
from rose_cli.responses.list_chains import list_chains
from rose_cli.responses.retrieve import retrieve_response

app = typer.Typer()

app.command(name="create")(create_response)
app.command(name="retrieve")(retrieve_response)
app.command(name="chains")(list_chains)
