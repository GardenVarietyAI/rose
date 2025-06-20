import os

from openai import OpenAI
from rich.console import Console

console = Console()

BASE_URL = os.environ.get("ROSE_BASE_URL", "http://localhost:8004/v1")
API_KEY = os.environ.get("ROSE_API_KEY", "sk-dummy-key")


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=300.0)
