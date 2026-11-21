"""Persistent semantic memory with Mem0 support and a local JSON fallback."""
import json, re
from pathlib import Path
from threading import RLock

class LocalMemory:
    def __init__(self, path: str):
        self.path, self.lock = Path(path), RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text("{}", encoding="utf-8")
    def _load(self): return json.loads(self.path.read_text(encoding="utf-8"))
    def _save(self, data): self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    def add(self, messages: str, user_id: str, session_id: str):
        facts = []
        text = messages.lower()
        if "python" in text and any(x in text for x in ("work in", "use", "prefer", "coding", "developer")):
            facts.append("User works with or prefers Python.")
        if "brief" in text or "concise" in text or "short responses" in text:
            facts.append("User prefers brief, concise responses.")
        if "code-only" in text or "only code" in text:
            facts.append("User prefers code-only responses when requesting code.")
        if "javascript" in text: facts.append("User works with JavaScript.")
        with self.lock:
            data = self._load(); existing = data.setdefault(user_id, [])
            for fact in facts:
                if not any(x["memory"] == fact for x in existing):
                    existing.append({"memory": fact, "session_id": session_id})
            self._save(data)
        return facts
    def search(self, query: str, user_id: str):
        with self.lock: return self._load().get(user_id, [])
    def get_all(self, user_id: str):
        with self.lock: return self._load().get(user_id, [])

class Mem0Memory:
    def __init__(self, api_key: str):
        from mem0 import MemoryClient
        self.client = MemoryClient(api_key=api_key)
    def add(self, messages: str, user_id: str, session_id: str):
        self.client.add(messages, user_id=user_id, metadata={"session_id": session_id})
        return []
    def search(self, query: str, user_id: str):
        result = self.client.search(query, user_id=user_id)
        return result.get("results", result) if isinstance(result, dict) else result
    def get_all(self, user_id: str):
        result = self.client.get_all(user_id=user_id)
        return result.get("results", result) if isinstance(result, dict) else result
