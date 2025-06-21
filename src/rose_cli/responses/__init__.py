import typer

from .create import create_response
from .retrieve import retrieve_response

app = typer.Typer()

app.command(name="create")(create_response)
app.command(name="retrieve")(retrieve_response)
