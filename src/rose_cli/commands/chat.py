from typing import Optional

import typer

from ..utils import get_client, get_endpoint_url

app = typer.Typer()


def chat_handler(
    message: str,
    model: str,
    url: Optional[str],
    local: bool,
    system: Optional[str],
    stream: bool,
    completion: bool = False,
):
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        if completion:
            if stream:
                response = client.completions.create(
                    model=model,
                    prompt=message,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].text:
                        print(chunk.choices[0].text, end="")
                print()
            else:
                response = client.completions.create(
                    model=model,
                    prompt=message,
                )
                print(response.choices[0].text)
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})
            if stream:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="")
                print()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                print(response.choices[0].message.content)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    message: str = typer.Argument(None, help="Message to send"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to use"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    stream: bool = typer.Option(False, "--stream", help="Stream response"),
    completion: bool = typer.Option(False, "--completion", "-c", help="Use completion mode instead of chat"),
):
    if ctx.invoked_subcommand is None and message:
        chat_handler(
            message=message, model=model, url=url, local=local, system=system, stream=stream, completion=completion
        )
