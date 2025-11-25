"""
Configuration settings for NeendAI
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/neendai"
    redis_url: str = "redis://localhost:6379"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # AWS (optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket: str = "neendai-audio"
    sns_topic_arn: Optional[str] = None

    # App settings
    alert_threshold: int = 30
    audio_retention_days: int = 30
    secret_key: str = "change-me-in-production"

    # Audio processing
    sample_rate: int = 16000
    segment_duration: float = 30.0

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
