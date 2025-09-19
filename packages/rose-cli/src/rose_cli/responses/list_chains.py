"""Seed default models into the database."""

from rich import print

from rose_cli.utils import console, get_client


def list_chains() -> None:
    """Seed default models into the database."""
    client = get_client()

    # Build auth headers from the client's API key
    headers = {}
    if client.api_key:
        headers["Authorization"] = f"Bearer {client.api_key}"

    try:
        response = client._client.get("/responses/chains", headers=headers)
        response.raise_for_status()
        result = response.json()
        for chain_id in result:
            print(chain_id)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
