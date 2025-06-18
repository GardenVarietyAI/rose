import typer
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from ...utils import get_client


def retrieve_response(
    response_id: str = typer.Argument(..., help="Response ID to retrieve"),
):
    """Retrieve a stored response."""
    client = get_client()
    try:
        response = client.responses.retrieve(response_id)
        print(f"Response ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Created: {response.created_at}")
        if response.usage:
            print(
                f"Usage: input_tokens={response.usage.input_tokens}, "
                f"output_tokens={response.usage.output_tokens}, "
                f"total={response.usage.total_tokens}"
            )
        print("\nOutput:")

        if not response.output:
            print("\nNo output")

        for item in response.output:
            if not isinstance(item, ResponseOutputMessage):
                break

            if not item.content:
                break

            for content_item in item.content:
                if isinstance(content_item, ResponseOutputText) and content_item.text:
                    print(content_item.text)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
