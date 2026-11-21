from app.retrieval import Retrieved, reciprocal_rank_fusion, split_text

def doc(cid): return Retrieved(cid, "p-"+cid, "child "+cid, "parent "+cid)

def test_rrf_merges_ranked_lists_and_promotes_overlap():
    dense, sparse = [doc("a"), doc("b")], [doc("b"), doc("c")]
    merged = reciprocal_rank_fusion(dense, sparse, k=60, limit=10)
    assert merged[0].child_id == "b"
    assert {x.child_id for x in merged} == {"a", "b", "c"}
    assert merged[0].score > merged[1].score

def test_rrf_respects_limit():
    assert len(reciprocal_rank_fusion([doc(str(i)) for i in range(8)], limit=3)) == 3

def test_split_text_empty_and_overlap():
    assert split_text("   ") == []
    assert len(split_text("x"*500, size=200, overlap=20)) == 3
