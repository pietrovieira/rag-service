from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Microservice"
    app_version: str = "0.1.0"
    debug: bool = False

    # RAG params
    embedding_dim: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    # Vector DB: in_memory | pgvector | qdrant | chroma
    vector_db_type: str = "pgvector"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # LLM: mock | openai
    llm_type: str = "mock"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Server
    host: str = "0.0.0.0"
    port: int = 9876

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
