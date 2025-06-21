import typer

from rose_cli.utils import get_client

app = typer.Typer()


@app.command()
def list_files():
    """List uploaded files."""
    client = get_client()
    try:
        files = client.files.list()
        for file in files.data:
            size = getattr(file, "bytes", 0)
            print(f"{file.id}\t{file.filename}\t{file.purpose}\t{size}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
