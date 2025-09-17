import typer

from rose_cli.rerank.score import score
from rose_cli.rerank.test import test

app = typer.Typer()

app.command(name="score")(score)
app.command(name="test")(test)
