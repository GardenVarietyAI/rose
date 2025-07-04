"""Auth commands for ROSE CLI."""

import typer

from rose_cli.auth.generate import generate_token

app = typer.Typer(help="Authentication management")

app.command(name="generate-token")(generate_token)
