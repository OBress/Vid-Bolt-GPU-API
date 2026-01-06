"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Mock Mode
    mock_mode: bool = True

    # API Authentication
    api_key: str = ""

    # ComfyUI Settings (future use)
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188

    # Limits
    max_image_size_mb: int = 10
    max_video_duration_seconds: int = 8

    # Logging
    log_level: str = "INFO"

    # CORS - comma-separated list of allowed origins
    cors_allowed_origins: str = "http://localhost:3000"

    @property
    def max_image_size_bytes(self) -> int:
        """Get max image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
