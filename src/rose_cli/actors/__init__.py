import typer

from rose_cli.actors.calculator_actor import calculator
from rose_cli.actors.file_reader_actor import file_reader

app = typer.Typer()

app.command(name="calculator-actor")(calculator)
app.command(name="file-reader")(file_reader)
