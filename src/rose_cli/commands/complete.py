"""Completion command for prompt completions."""
from typing import Optional
import typer
from ..utils import get_client, get_endpoint_url
app = typer.Typer()
@app.command()

def complete(
    prompt: str = typer.Argument(..., help="Prompt to complete"),
    model: str = typer.Option("qwen-coder", "--model", "-m", help="Model to use"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Base URL"),
    local: bool = typer.Option(True, "--local/--remote", "-l", help="Use local service"),
    max_tokens: int = typer.Option(100, "--max-tokens", "-t", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream response"),
    echo: bool = typer.Option(False, "--echo", "-e", help="Echo the prompt in the response"),
):
    """Generate text completions from prompts."""
    endpoint_url = get_endpoint_url(url, local)
    client = get_client(endpoint_url)
    try:
        if stream:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                echo=echo,
            )
            for chunk in response:
                if chunk.choices[0].text:
                    print(chunk.choices[0].text, end="", flush=True)
            print()
        else:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=echo,
            )
            print(response.choices[0].text)
    except Exception as e:
        print(f"error: {e}", file=typer.get_text_stream("stderr"))