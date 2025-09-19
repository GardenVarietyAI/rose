import os

import cohere
from openai import AsyncOpenAI, OpenAI
from rich.console import Console

console = Console()

BASE_URL_OPENAI = os.environ.get("ROSE_BASE_URL_OPENAI", "http://localhost:8004/v1")
BASE_URL_COHERE = os.environ.get("ROSE_BASE_URL_COHERE", "http://localhost:8004")
API_KEY = os.environ.get("ROSE_API_KEY") or "sk-dummy-key"


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL_OPENAI, api_key=API_KEY, timeout=300.0)


def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL_OPENAI, api_key=API_KEY, timeout=300.0)


def get_cohere_client() -> cohere.Client:
    return cohere.Client(api_key=API_KEY, base_url=BASE_URL_COHERE)


def get_cohere_async_client() -> cohere.AsyncClient:
    return cohere.AsyncClient(api_key=API_KEY, base_url=BASE_URL_COHERE)
