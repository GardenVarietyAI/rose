from typing import Optional

import typer

from ...utils import get_client, get_endpoint_url

app = typer.Typer()


@app.command()
def get_file(
    file_id: str = typer.Argument(..., help="File ID"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Get file details."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        file = client.files.retrieve(file_id)
        print(f"ID: {file.id}")
        print(f"Filename: {file.filename}")
        print(f"Purpose: {file.purpose}")
        print(f"Size: {getattr(file, 'bytes', 0)}")
        print(f"Created: {file.created_at}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
