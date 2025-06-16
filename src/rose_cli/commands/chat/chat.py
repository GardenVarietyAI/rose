from typing import Optional

import typer
from openai import OpenAI

from ...utils import get_client, get_endpoint_url


def _do_chat(client: OpenAI, model: str, prompt: str, system: Optional[str], stream: bool):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        if stream:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}", file=typer.get_text_stream("stderr"))


def chat(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Message to send"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to use"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response"),
):
    """Chat with models using a single message."""
    if ctx.invoked_subcommand is not None:
        return

    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    _do_chat(client, model, prompt, system, stream)
