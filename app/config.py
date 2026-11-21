from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str = ""
    cohere_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "rag_children"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    cohere_rerank_model: str = "rerank-english-v3.0"
    rerank_min_score: float = 0.0
    mem0_api_key: str = ""
    memory_backend: str = "local"
    data_dir: str = "./data"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

settings = Settings()
