import typer

from rose_cli.finetune.cancel import cancel_job
from rose_cli.finetune.create import create_job
from rose_cli.finetune.events import show_events
from rose_cli.finetune.get import get_job
from rose_cli.finetune.list import list_jobs
from rose_cli.finetune.metrics import show_metrics
from rose_cli.finetune.pause import pause_job
from rose_cli.finetune.resume import resume_job

app = typer.Typer()

app.command(name="list")(list_jobs)
app.command(name="create")(create_job)
app.command(name="get")(get_job)
app.command(name="events")(show_events)
app.command(name="metrics")(show_metrics)
app.command(name="pause")(pause_job)
app.command(name="resume")(resume_job)
app.command(name="cancel")(cancel_job)
