import os

import typer

from rose_cli.utils import get_client

app = typer.Typer()


@app.command()
def upload_file(
    file_path: str = typer.Argument(..., help="Path to file"),
    purpose: str = typer.Option(
        "fine-tune", "--purpose", "-p", help="File purpose (assistants, batch, fine-tune, vision, user_data, evals)"
    ),
):
    """Upload a file."""
    client = get_client()
    try:
        if not os.path.exists(file_path):
            print(f"error: file not found: {file_path}", file=typer.get_text_stream("stderr"))
            raise typer.Exit(1)
        with open(file_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose=purpose)  # type: ignore
        print(file_obj.id)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
