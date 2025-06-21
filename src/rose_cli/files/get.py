import typer

from rose_cli.utils import get_client

app = typer.Typer()


@app.command()
def get_file(
    file_id: str = typer.Argument(..., help="File ID"),
):
    """Get file details."""
    client = get_client()
    try:
        file = client.files.retrieve(file_id)
        print(f"ID: {file.id}")
        print(f"Filename: {file.filename}")
        print(f"Purpose: {file.purpose}")
        print(f"Size: {getattr(file, 'bytes', 0)}")
        print(f"Created: {file.created_at}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
