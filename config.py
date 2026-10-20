from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_core.globals import set_debug


PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    pgvector_url: str = Field(validation_alias="PGVECTOR_URL")
    database_url: str = Field(validation_alias="DATABASE_URL")
    langchain_debug: bool = Field(default=False, validation_alias="LANGCHAIN_DEBUG")


settings = Settings()


def configure_langchain() -> None:
    """Apply shared LangChain runtime options for standalone scripts."""
    set_debug(settings.langchain_debug)
