"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration; secrets are never serialized by the API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PRODUCT_DISCOVERY_",
        extra="ignore",
        frozen=True,
    )

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-6-astra"
    catalogue_path: Path = Path("data/processed/laptop_product_database_v3.parquet")
    max_candidates: int = Field(default=5, ge=1, le=20)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide immutable-by-convention settings instance."""

    return Settings()
