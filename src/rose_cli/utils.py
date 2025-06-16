from typing import Optional

from openai import OpenAI
from rich.console import Console

console = Console()


def get_client(base_url: Optional[str] = None) -> OpenAI:
    url = base_url or "http://localhost:8004/v1"
    return OpenAI(base_url=url, api_key="dummy-key", timeout=300.0)


def get_endpoint_url(url: Optional[str] = None, local: bool = True) -> Optional[str]:
    if local:
        return "http://localhost:8004/v1"
    return url
