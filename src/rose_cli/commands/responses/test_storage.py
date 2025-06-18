import typer

from ...utils import get_client


def test_storage():
    """Test response storage functionality."""
    client = get_client()
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
        print(response.output_text)
    except Exception as e:
        print(f"[red]Storage test failed: {e}[/red]", file=typer.get_text_stream("stderr"))
