import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("SETTINGS_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llama_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias=AliasChoices("LLAMA_BASE_URL", "OPENAI_BASE_URL"),
    )
    llama_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLAMA_API_KEY", "OPENAI_API_KEY"),
    )
    llama_model_path: str = Field(default="", validation_alias="LLAMA_MODEL_PATH")
    nltk_data: str = Field(default="./vendor/nltk_data", validation_alias="NLTK_DATA")
