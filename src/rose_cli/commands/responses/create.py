from typing import Optional

import typer

from ...utils import get_client, get_endpoint_url


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
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    input_data = [{"type": "message", "role": "user", "content": message}]
    try:
        response = client.responses.create(
            model=model,
            input=input_data,
            instructions=instructions,
            store=store,
            stream=stream,
        )
        if stream:
            for event in response:
                if hasattr(event, "delta") and event.delta:
                    print(event.delta, end="", flush=True)
            print()
        else:
            print(f"Response ID: {response.id}")
            print(f"Status: {response.status}")
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            print(content.text)
                        elif hasattr(content, "content"):
                            print(content.content)
                elif hasattr(item, "text"):
                    print(item.text)
                elif isinstance(item, dict) and "content" in item:
                    print(item["content"])
            if store:
                print(f"\nResponse stored with ID: {response.id}")
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
