from types import SimpleNamespace
from app.rerank import CohereReranker
from app.retrieval import Retrieved

def test_reranker_filters_low_scores_and_limits_top_n():
    docs = [Retrieved(str(i), "p", "child", f"document {i}") for i in range(3)]
    reranker = CohereReranker(min_score=0.5)
    class FakeClient:
        def rerank(self, **kwargs):
            return SimpleNamespace(results=[SimpleNamespace(index=2,relevance_score=.9), SimpleNamespace(index=0,relevance_score=.2)])
    reranker.client = FakeClient()
    result = reranker.rerank("query", docs, top_n=2)
    assert len(result) == 1
    assert result[0][0].child_id == "2"
