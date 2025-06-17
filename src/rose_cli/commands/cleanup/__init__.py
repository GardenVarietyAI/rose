import typer

from .all import cleanup_all
from .files import cleanup_files
from .jobs import cleanup_jobs
from .models import cleanup_models

app = typer.Typer()

app.command(name="models")(cleanup_models)
app.command(name="files")(cleanup_files)
app.command(name="jobs")(cleanup_jobs)
app.command(name="all")(cleanup_all)
