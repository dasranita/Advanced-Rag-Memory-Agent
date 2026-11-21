"""Cohere reranking with a deterministic local fallback for development/tests."""
class CohereReranker:
    def __init__(self, api_key: str = "", model: str = "rerank-english-v3.0", min_score: float = 0.0):
        self.api_key, self.model, self.min_score = api_key, model, min_score
        self.client = None
        if api_key:
            import cohere
            self.client = cohere.ClientV2(api_key=api_key)
    def rerank(self, query, documents, top_n=5):
        if not documents: return []
        if not self.client:
            terms = set(query.lower().split())
            scored = [(d, len(terms & set(d.parent_text.lower().split())) / max(len(terms), 1)) for d in documents]
            return [(d, score) for d, score in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n] if score >= self.min_score]
        response = self.client.rerank(model=self.model, query=query, documents=[d.parent_text for d in documents], top_n=min(top_n, len(documents)))
        output = []
        for result in response.results:
            score = float(result.relevance_score)
            if score >= self.min_score:
                output.append((documents[result.index], score))
        return output
