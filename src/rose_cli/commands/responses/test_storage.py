from typing import Optional

import typer

from ...utils import get_client, get_endpoint_url


def test_storage(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Test response storage functionality."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        print("Testing response storage...")

        # Create a test response with storage
        test_message = "This is a test message for storage functionality"
        response = client.responses.create(
            model="qwen-coder",
            input=[{"type": "message", "role": "user", "content": test_message}],
            instructions="You are a helpful assistant. Acknowledge the test message.",
            store=True,
        )

        response_id = response.id
        print(f"✓ Created response with ID: {response_id}")

        # Try to retrieve it
        import httpx

        with httpx.Client() as http_client:
            retrieve_response = http_client.get(f"{endpoint_url}/responses/{response_id}")
            retrieve_response.raise_for_status()
            data = retrieve_response.json()
            print(f"✓ Successfully retrieved response {data['id']}")
            print(f"  Status: {data['status']}")
            print(f"  Created: {data['created']}")

        print("\n[green]Storage test passed![/green]")

    except Exception as e:
        print(f"[red]Storage test failed: {e}[/red]", file=typer.get_text_stream("stderr"))
