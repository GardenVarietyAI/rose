import typer

from .get import get_model
from .list import list_models
from .pull import pull_model

app = typer.Typer()

app.command(name="list")(list_models)
app.command(name="get")(get_model)
app.command(name="pull")(pull_model)
