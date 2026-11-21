from time import perf_counter
from app.config import settings
from app.retrieval import HybridRetriever, InMemoryHybridRetriever, OpenAIEmbedder
from app.rerank import CohereReranker
from app.memory import LocalMemory, Mem0Memory

class RAGAgent:
    def __init__(self):
        self.retriever = None
        if settings.openai_api_key:
            try:
                self.retriever = HybridRetriever(OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model), settings.qdrant_path, settings.qdrant_collection, settings.qdrant_url, settings.qdrant_api_key, f"{settings.data_dir}/index_metadata.json")
            except Exception as exc:
                print(f"Qdrant initialization failed; using in-memory retriever: {exc}")
        if self.retriever is None: self.retriever = InMemoryHybridRetriever()
        self.reranker = CohereReranker(settings.cohere_api_key, settings.cohere_rerank_model, settings.rerank_min_score)
        self.memory = Mem0Memory(settings.mem0_api_key) if settings.memory_backend.lower() == "mem0" and settings.mem0_api_key else LocalMemory(f"{settings.data_dir}/memories.json")
        self.llm = None
        if settings.openai_api_key:
            from openai import OpenAI
            self.llm = OpenAI(api_key=settings.openai_api_key)
    def ingest(self, text, source="manual"): return self.retriever.add_text(text, source)
    def chat(self, query, user_id, session_id):
        started = perf_counter()
        t = perf_counter(); candidates = self.retriever.search(query, limit=20); retrieval_ms = (perf_counter()-t)*1000
        t = perf_counter(); ranked = self.reranker.rerank(query, candidates, top_n=5); rerank_ms = (perf_counter()-t)*1000
        memories = self.memory.search(query, user_id)
        profile = "\n".join(str(x.get("memory", x)) if isinstance(x, dict) else str(x) for x in memories)
        context = "\n\n".join(f"[{i+1}] {doc.parent_text}" for i,(doc,_) in enumerate(ranked))
        if self.llm:
            system = "You are a helpful retrieval-augmented assistant. Answer using supplied context when relevant. If context is insufficient, say so. Never invent citations."
            if profile: system += f"\nUser profile (persistent preferences; follow when relevant):\n{profile}"
            if "code-only" in profile.lower() and any(k in query.lower() for k in ("write", "code", "implement", "function", "program")):
                system += "\nFor this coding request, output code only, with no explanatory prose."
            prompt = f"Question: {query}\n\nRetrieved context:\n{context or 'No relevant documents found.'}"
            t = perf_counter(); result = self.llm.chat.completions.create(model=settings.openai_chat_model, messages=[{"role":"system","content":system},{"role":"user","content":prompt}], temperature=0.2); answer = result.choices[0].message.content or ""
            generation_ms = (perf_counter()-t)*1000
        else:
            generation_ms = 0.0
            answer = ("Demo mode: configure OPENAI_API_KEY to enable LLM answers.\n\n" + context) if context else "Demo mode is active. Add documents with POST /api/ingest and configure OPENAI_API_KEY for generated answers."
        # Save interaction after generation so the current answer uses the existing profile.
        try: self.memory.add(f"User: {query}\nAssistant: {answer}", user_id, session_id)
        except Exception as exc: print(f"Memory write failed: {exc}")
        total_ms = (perf_counter()-started)*1000
        return {"answer": answer, "sources": [{"source": d.source, "parent_id": d.parent_id, "relevance_score": score, "text": d.parent_text[:500]} for d,score in ranked], "telemetry": {"retrieval_latency_ms": round(retrieval_ms,3), "rerank_latency_ms": round(rerank_ms,3), "generation_latency_ms": round(generation_ms,3), "total_latency_ms": round(total_ms,3)}}
