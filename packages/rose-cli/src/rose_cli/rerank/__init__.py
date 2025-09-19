import typer

from rose_cli.rerank.score import score

app = typer.Typer()

app.command(name="score")(score)
