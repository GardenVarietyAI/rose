import typer

from rose_cli.models.add import add_model
from rose_cli.models.download import download_model
from rose_cli.models.get import get_model
from rose_cli.models.list import list_models
from rose_cli.models.seed import seed_models

app = typer.Typer()

app.command(name="list")(list_models)
app.command(name="get")(get_model)
app.command(name="download")(download_model)
app.command(name="add")(add_model)
app.command(name="seed")(seed_models)
