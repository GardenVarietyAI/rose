from typing import Optional

import typer
from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseTextDeltaEvent

from ...utils import get_client


def create_response(
    message: str = typer.Argument(..., help="Message to send"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to use"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    store: bool = typer.Option(True, "--store/--no-store", help="Store response for later retrieval"),
    stream: bool = typer.Option(False, "--stream", help="Stream response"),
    instructions: Optional[str] = typer.Option(None, "--instructions", "-i", help="System instructions"),
):
    """Create a response using the Responses API."""
    client = get_client()
    try:
        if stream:
            response_stream = client.responses.create(
                model=model,
                input=message,
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
                input=message,
                instructions=instructions,
                store=store,
                stream=False,
            )
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
