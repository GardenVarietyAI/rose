import typer

from rose_cli.runs.create import create_run
from rose_cli.runs.get import get_run
from rose_cli.runs.list import list_runs

app = typer.Typer()
app.command("create")(create_run)
app.command("get")(get_run)
app.command("list")(list_runs)
