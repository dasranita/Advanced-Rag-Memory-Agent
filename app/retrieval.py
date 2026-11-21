"""Parent/child indexing, dense+sparse retrieval and reciprocal-rank fusion."""
from __future__ import annotations
import re, uuid, json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

@dataclass
class Retrieved:
    child_id: str
    parent_id: str
    text: str
    parent_text: str
    source: str = "unknown"
    score: float = 0.0

class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...

class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client, self.model = OpenAI(api_key=api_key), model
    def embed_documents(self, texts):
        if not texts: return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [x.embedding for x in response.data]
    def embed_query(self, text): return self.embed_documents([text])[0]

def split_text(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    if overlap >= size: raise ValueError("overlap must be smaller than size")
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

def reciprocal_rank_fusion(*ranked_lists: list[Retrieved], k: int = 60, limit: int = 20) -> list[Retrieved]:
    """Merge ranked lists by child id; scores are sums of 1/(k + rank)."""
    scores: dict[str, float] = {}
    docs: dict[str, Retrieved] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc.child_id] = scores.get(doc.child_id, 0.0) + 1.0 / (k + rank)
            docs[doc.child_id] = doc
    merged = []
    for child_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
        doc = docs[child_id]
        merged.append(Retrieved(doc.child_id, doc.parent_id, doc.text, doc.parent_text, doc.source, score))
    return merged

class HybridRetriever:
    """Qdrant dense index + BM25 sparse index. Qdrant can run persistently on disk."""
    def __init__(self, embedder: Embedder, path: str, collection: str, url: str = "", api_key: str = "", metadata_path: str = "./data/index_metadata.json"):
        from qdrant_client import QdrantClient, models
        from rank_bm25 import BM25Okapi
        self.models, self.embedder, self.collection = models, embedder, collection
        self.client = QdrantClient(url=url, api_key=api_key or None) if url else QdrantClient(path=path)
        self.parents: dict[str, dict] = {}
        self.children: list[dict] = []
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            saved = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.parents, self.children = saved.get("parents", {}), saved.get("children", [])
        self.bm25 = None
        self._BM25Okapi = BM25Okapi
        self._ensure_collection()
        self._rebuild_bm25()
    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            # OpenAI text-embedding-3-small produces 1536 dimensions.
            self.client.create_collection(self.collection, vectors_config=self.models.VectorParams(size=1536, distance=self.models.Distance.COSINE))
    def add_text(self, text: str, source: str = "manual", parent_size: int = 1000, child_size: int = 200):
        from qdrant_client.models import PointStruct
        parents = split_text(text, parent_size, min(100, parent_size-1))
        new_children = []
        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            self.parents[parent_id] = {"text": parent_text, "source": source}
            for child_text in split_text(parent_text, child_size, min(30, child_size-1)):
                item = {"id": str(uuid.uuid4()), "parent_id": parent_id, "text": child_text, "source": source}
                new_children.append(item)
        if not new_children: return 0
        vectors = self.embedder.embed_documents([x["text"] for x in new_children])
        points = [PointStruct(id=x["id"], vector=v, payload=x) for x, v in zip(new_children, vectors)]
        self.client.upsert(collection_name=self.collection, points=points)
        self.children.extend(new_children)
        self._rebuild_bm25()
        self.metadata_path.write_text(json.dumps({"parents": self.parents, "children": self.children}, indent=2), encoding="utf-8")
        return len(new_children)
    def _rebuild_bm25(self):
        self.bm25 = self._BM25Okapi([self._tokens(x["text"]) for x in self.children]) if self.children else None
    @staticmethod
    def _tokens(text): return re.findall(r"\w+", text.lower())
    def search(self, query: str, limit: int = 20) -> list[Retrieved]:
        from qdrant_client.models import query as qmodels
        dense_hits = self.client.search(collection_name=self.collection, query_vector=self.embedder.embed_query(query), limit=limit, with_payload=True) if self.children else []
        dense = [self._to_retrieved(hit.payload, hit.score) for hit in dense_hits]
        sparse = []
        if self.bm25:
            scores = self.bm25.get_scores(self._tokens(query))
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
            sparse = [self._to_retrieved(self.children[i], float(scores[i])) for i in indices if scores[i] > 0]
        return reciprocal_rank_fusion(dense, sparse, limit=limit)
    def _to_retrieved(self, payload, score):
        parent = self.parents.get(payload["parent_id"], {})
        return Retrieved(payload["id"], payload["parent_id"], payload["text"], parent.get("text", payload["text"]), payload.get("source", "unknown"), float(score))

class InMemoryHybridRetriever:
    """Deterministic lightweight retriever used by tests and demo mode."""
    def __init__(self): self.docs: list[Retrieved] = []
    def add_text(self, text: str, source: str = "manual", **kwargs):
        count = 0
        for parent in split_text(text, 1000, 100):
            pid = str(uuid.uuid4())
            for child in split_text(parent, 200, 30):
                self.docs.append(Retrieved(str(uuid.uuid4()), pid, child, parent, source)); count += 1
        return count
    def search(self, query: str, limit: int = 20):
        terms = set(HybridRetriever._tokens(query))
        ranked = sorted(self.docs, key=lambda d: len(terms & set(HybridRetriever._tokens(d.text))), reverse=True)
        return [Retrieved(d.child_id,d.parent_id,d.text,d.parent_text,d.source,float(len(terms & set(HybridRetriever._tokens(d.text))))) for d in ranked[:limit] if terms & set(HybridRetriever._tokens(d.text))]
