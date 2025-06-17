import typer

from .create import create_eval
from .delete import delete_eval
from .get import get_eval
from .list import list_evals
from .run import run_eval

app = typer.Typer()

app.command(name="create")(create_eval)
app.command(name="list")(list_evals)
app.command(name="get")(get_eval)
app.command(name="run")(run_eval)
app.command(name="delete")(delete_eval)
