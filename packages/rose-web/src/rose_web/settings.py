from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8005
    openai_api_url: str = "http://localhost:8004/v1"
    openai_api_key: str = "sk-dummykey"
    default_model: str = "Qwen--Qwen3-1.7B-GGUF"
    data_dir: str = "./data"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
