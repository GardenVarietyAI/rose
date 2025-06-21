import typer

from .create import create_run
from .get import get_run
from .list import list_runs

app = typer.Typer()
app.command("create")(create_run)
app.command("get")(get_run)
app.command("list")(list_runs)
