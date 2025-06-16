"""Compare command."""

from typing import Optional

import typer

from ..utils import get_client

app = typer.Typer()


@app.command()
def models(
    message: str = typer.Argument(..., help="Message to test"),
    local_model: str = typer.Option("qwen-coder", "--local-model", help="Local model"),
    remote_model: str = typer.Option("gpt-4o", "--remote-model", help="Remote model"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
):
    """Compare local vs remote model responses."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    try:
        local_client = get_client("http://localhost:8004/v1")
        local_response = local_client.chat.completions.create(
            model=local_model,
            messages=messages,
        )
        print("LOCAL:")
        print(local_response.choices[0].message.content)
        print()
    except Exception as e:
        print(f"local error: {e}", file=typer.get_text_stream("stderr"))
    try:
        remote_client = get_client()
        remote_response = remote_client.chat.completions.create(
            model=remote_model,
            messages=messages,
        )
        print("REMOTE:")
        print(remote_response.choices[0].message.content)
    except Exception as e:
        print(f"remote error: {e}", file=typer.get_text_stream("stderr"))
