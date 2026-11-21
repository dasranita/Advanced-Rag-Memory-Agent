from fastapi import FastAPI, HTTPException
from app.agent import RAGAgent
from app.schemas import ChatRequest, ChatResponse, MemoryResponse, IngestRequest

app = FastAPI(title="Advanced RAG Memory Agent", version="1.0.0", description="Hybrid RAG with Cohere reranking and persistent semantic memory")
agent = RAGAgent()

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        result = agent.chat(payload.query, payload.user_id, payload.session_id)
        return {**result, "user_id": payload.user_id, "session_id": payload.session_id}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/users/{user_id}/memories", response_model=MemoryResponse)
def get_memories(user_id: str):
    try: return {"user_id": user_id, "memories": agent.memory.get_all(user_id)}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/ingest")
def ingest(payload: IngestRequest):
    try: return {"indexed_children": agent.ingest(payload.text, payload.source)}
    except Exception as exc: raise HTTPException(status_code=500, detail=str(exc))
