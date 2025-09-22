from dataclasses import dataclass


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8005
    openai_api_url: str = "http://localhost:8004/v1"
    opneai_api_key = "sk-dummykey"


settings = Settings()
