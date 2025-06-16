"""Upload command."""

import os
from typing import Optional

import typer

from ..utils import get_client, get_endpoint_url

app = typer.Typer()


@app.command()
def file(
    file_path: str = typer.Argument(..., help="Path to file"),
    purpose: str = typer.Option("fine-tune", "--purpose", "-p", help="File purpose"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Upload a file."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        if not os.path.exists(file_path):
            print(f"error: file not found: {file_path}", file=typer.get_text_stream("stderr"))
            raise typer.Exit(1)
        with open(file_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose=purpose)
        print(file_obj.id)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
