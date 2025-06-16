from typing import Optional

import typer

from ...utils import get_client, get_endpoint_url


def retrieve_response(
    response_id: str = typer.Argument(..., help="Response ID to retrieve"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Retrieve a stored response."""
    endpoint_url = get_endpoint_url(url, local)
    get_client(endpoint_url)
    try:
        # Use custom endpoint for retrieval
        import httpx

        with httpx.Client() as http_client:
            response = http_client.get(f"{endpoint_url}/responses/{response_id}")
            response.raise_for_status()
            data = response.json()
            print(f"Response ID: {data['id']}")
            print(f"Status: {data['status']}")
            print(f"Created: {data['created']}")
            print("\nOutput:")
            for item in data["output"]:
                if isinstance(item, dict):
                    if "content" in item:
                        print(item["content"])
                    elif "type" in item and item["type"] == "message":
                        print(f"[{item.get('role', 'assistant')}]: {item.get('content', '')}")
                else:
                    print(item)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.text}", file=typer.get_text_stream("stderr"))
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
