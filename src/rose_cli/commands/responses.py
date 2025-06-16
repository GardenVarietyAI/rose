"""Responses API command."""
from typing import Optional
import typer
from ..utils import get_client, get_endpoint_url
app = typer.Typer()
@app.command()

def create(
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
            if store:
                print(f"\n✓ Response stored with ID: {response.id}")
                print("Use 'rose responses retrieve {id}' to get it later")
    except Exception as e:
        print(f"Error: {e}", file=typer.get_text_stream("stderr"))
@app.command()

def retrieve(
    response_id: str = typer.Argument(..., help="Response ID to retrieve"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Retrieve a stored response by ID."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        response = client.responses.retrieve(response_id)
        if hasattr(response, "error") or response is None:
            print(f"Error: Response {response_id} not found", file=typer.get_text_stream("stderr"))
            return
        print(f"Retrieved Response ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Model: {response.model}")
        for item in response.output:
            if hasattr(item, "content"):
                for content in item.content:
                    if hasattr(content, "text"):
                        print(content.text)
                    elif hasattr(content, "content"):
                        print(content.content)
            elif hasattr(item, "text"):
                print(item.text)
    except Exception as e:
        print(f"Error: {e}", file=typer.get_text_stream("stderr"))
@app.command()

def test_storage():
    """Test responses storage functionality."""
    endpoint_url = get_endpoint_url(None, True)
    client = get_client(endpoint_url)
    print("Testing responses storage...")
    print("\n1. Testing store=True...")
    response = client.responses.create(
        model="qwen-coder",
        input=[{"type": "message", "role": "user", "content": "I am planning to build a new feature"}],
        store=True,
        stream=False,
    )
    print(f"Created response: {response.id}")
    try:
        retrieved = client.responses.retrieve(response.id)
        print(f"✓ Successfully retrieved: {retrieved.id}")
    except Exception as e:
        print(f"✗ Failed to retrieve: {e}")
    print("\n2. Testing store=False...")
    response2 = client.responses.create(
        model="qwen-coder",
        input=[{"type": "message", "role": "user", "content": "This should not be stored"}],
        store=False,
        stream=False,
    )
    print(f"Created response: {response2.id}")
    try:
        retrieved2 = client.responses.retrieve(response2.id)
        if retrieved2 is None or hasattr(retrieved2, "error"):
            print(f"✓ Correctly failed to retrieve non-stored response")
        else:
            print(f"✗ Unexpectedly retrieved: {retrieved2.id}")
    except Exception as e:
        print(f"✓ Correctly failed to retrieve: {e}")
    print("\nStorage test complete!")