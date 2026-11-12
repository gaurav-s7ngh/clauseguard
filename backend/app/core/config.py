from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

_project_root = Path(__file__).resolve().parent.parent.parent.parent
_default_db_url = f"sqlite:///{(_project_root / 'clauseguard.db').as_posix()}"


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ClauseGuard API"
    VERSION: str = "1.0.0"
    
    # Database URL: defaults to local absolute SQLite for zero-config local runs,
    # or can be set via env var to postgresql+psycopg://user:password@localhost:5432/clauseguard
    DATABASE_URL: str = _default_db_url
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    
    # Upload limits
    MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    
    # Model pipeline version identifier
    PIPELINE_VERSION: str = "clauseguard-stage6-v1.0"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()

