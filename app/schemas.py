from pydantic import BaseModel, Field
from typing import Any

class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

class Telemetry(BaseModel):
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

class ChatResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str
    sources: list[dict[str, Any]] = []
    telemetry: Telemetry

class MemoryResponse(BaseModel):
    user_id: str
    memories: list[dict[str, Any]]

class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = "manual"
