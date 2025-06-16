import typer

from .cancel import cancel_job
from .create import create_job
from .events import show_events
from .get import get_job
from .list import list_jobs
from .pause import pause_job
from .resume import resume_job
from .test import test_fine_tuned_model

app = typer.Typer()

app.command(name="list")(list_jobs)
app.command(name="create")(create_job)
app.command(name="get")(get_job)
app.command(name="events")(show_events)
app.command(name="pause")(pause_job)
app.command(name="resume")(resume_job)
app.command(name="cancel")(cancel_job)
app.command(name="test")(test_fine_tuned_model)
