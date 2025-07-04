import typer

from rose_cli.actors.calculator.command import calculator
from rose_cli.actors.code_reviewer.command import code_reviewer
from rose_cli.actors.file_reader.command import file_reader

app = typer.Typer()

app.command(name="calculator")(calculator)
app.command(name="code-reviewer")(code_reviewer)
app.command(name="file-reader")(file_reader)
