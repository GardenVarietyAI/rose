from typing import Optional

import typer
from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseTextDeltaEvent

from rose_cli.utils import get_client


def create_response(
    message: str = typer.Argument(..., help="Message to send"),
    model: str = typer.Option("Qwen--Qwen2.5-1.5B-Instruct", "--model", "-m", help="Model to use"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store response for later retrieval"),
    stream: bool = typer.Option(False, "--stream", help="Stream response"),
    instructions: Optional[str] = typer.Option(None, "--instructions", "-i", help="System instructions"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only output the response ID (non-streaming only)"),
) -> None:
    """Create a response using the Responses API."""
    client = get_client()
    try:
        if stream:
            response_stream = client.responses.create(
                model=model,
                input=[{"role": "user", "content": message}],
                instructions=instructions,
                store=store,
                stream=True,
            )
            for event in response_stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    print(event.delta, end="", flush=True)
            print()
        else:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": message}],
                instructions=instructions,
                store=store,
                stream=False,
            )

            if quiet:
                if store:
                    print(response.id)
            else:
                print(f"Response ID: {response.id}")
                print(f"Status: {response.status}")
                if response.output:
                    for item in response.output:
                        if isinstance(item, ResponseOutputMessage) and item.content:
                            for content_item in item.content:
                                if isinstance(content_item, ResponseOutputText) and content_item.text:
                                    print(content_item.text)
                if store:
                    print(f"\nResponse stored with ID: {response.id}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
