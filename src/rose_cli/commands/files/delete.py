from typing import Optional

import typer

from ...utils import get_client, get_endpoint_url

app = typer.Typer()


@app.command()
def delete_file(
    file_id: str = typer.Argument(..., help="File ID to delete"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Delete a file."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        client.files.delete(file_id)
        print(f"File {file_id} deleted")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
