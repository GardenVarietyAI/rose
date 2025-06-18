import typer

from ...utils import get_client


def create_completion(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt to complete"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to use"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-t", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream response"),
    echo: bool = typer.Option(False, "--echo", "-e", help="Echo the prompt in the response"),
):
    """Generate text completions from prompts."""
    if ctx.invoked_subcommand is not None:
        return

    client = get_client()

    try:
        if stream:
            stream_response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                echo=echo,
            )
            for chunk in stream_response:
                if chunk.choices[0].text:
                    print(chunk.choices[0].text, end="", flush=True)
            print()
        else:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                echo=echo,
            )
            print(response.choices[0].text)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))
