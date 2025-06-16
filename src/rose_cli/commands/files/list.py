from typing import Optional

import typer

from ...utils import get_client, get_endpoint_url

app = typer.Typer()


@app.command()
def list_files(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """List uploaded files."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        files = client.files.list()
        for file in files.data:
            size = getattr(file, "bytes", 0)
            print(f"{file.id}\t{file.filename}\t{file.purpose}\t{size}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
