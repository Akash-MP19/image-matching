import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./image_matching.db"
    POSTGRES_DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/image_matching"
    SQLITE_FALLBACK: bool = True

    # AI Provider
    GEMINI_API_KEY: str = ""
    VISION_MODEL: str = "gemini-2.0-flash"
    EMBEDDING_MODEL: str = "text-embedding-004"
    OFFLINE_MODE: bool = True

    # Thresholds and Limits
    MAX_BATCH_RETRIES: int = 3
    BATCH_RETRY_DELAY_SEC: float = 1.0
    SIMILARITY_THRESHOLD: float = 0.55
    CONFIDENCE_THRESHOLD: float = 0.70
    BUDGET_LIMIT_USD: float = 5.00

    # Pricing rates (per 1,000 tokens or per image call)
    # Gemini 2.0 Flash vision: ~$0.00001875 per 258 image tokens ($0.075 / 1M input tokens)
    COST_PER_VISION_CALL_USD: float = 0.000025
    COST_PER_1K_EMBEDDING_TOKENS_USD: float = 0.00002

    # Paths
    IMAGE_CORPUS_DIR: str = str(BASE_DIR / "data" / "corpus")
    VISION_CACHE_FILE: str = str(BASE_DIR / "data" / "vision_cache.json")
    EVAL_POSTS_FILE: str = str(BASE_DIR / "data" / "eval_posts.json")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
