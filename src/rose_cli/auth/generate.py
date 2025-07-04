"""Generate authentication token."""

import uuid

import typer


def generate_token() -> None:
    """Generate a new authentication token."""
    token = str(uuid.uuid4())
    typer.echo(token)
