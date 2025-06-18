import typer

from .create import create_eval
from .list import list_evals
from .run import run_eval
from .status import eval_status

app = typer.Typer(name="evals", help="Manage evaluations")

app.command("create")(create_eval)
app.command("list")(list_evals)
app.command("run")(run_eval)
app.command("status")(eval_status)
