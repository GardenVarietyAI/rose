import typer

from .create import create_response
from .retrieve import retrieve_response
from .test_storage import test_storage

app = typer.Typer()

app.command(name="create")(create_response)
app.command(name="retrieve")(retrieve_response)
app.command(name="test-storage")(test_storage)
