from typing import Optional

import typer
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from ...utils import get_client, get_endpoint_url


def retrieve_response(
    response_id: str = typer.Argument(..., help="Response ID to retrieve"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
):
    """Retrieve a stored response."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        response = client.responses.retrieve(response_id)
        print(f"Response ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Created: {response.created_at}")
        print(
            f"Usage: input_tokens={response.usage.input_tokens}, "
            f"output_tokens={response.usage.output_tokens}, "
            f"total={response.usage.total_tokens}"
        )
        print("\nOutput:")
        if response.output:
            for item in response.output:
                if isinstance(item, ResponseOutputMessage) and item.content:
                    for content_item in item.content:
                        if isinstance(content_item, ResponseOutputText) and content_item.text:
                            print(content_item.text)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
