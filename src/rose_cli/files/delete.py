import typer

from rose_cli.utils import get_client

app = typer.Typer()


@app.command()
def delete_file(
    file_id: str = typer.Argument(..., help="File ID to delete"),
):
    """Delete a file."""
    client = get_client()
    try:
        client.files.delete(file_id)
        print(f"File {file_id} deleted")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
