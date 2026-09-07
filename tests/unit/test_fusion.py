"""RRF and weighted fusion — pure math, no services."""

from app.retrieval.fusion import rrf, weighted


def hits(source, ids_scores):
    return [{"chunk_id": cid, "score": s, "source": source} for cid, s in ids_scores]


def test_rrf_rewards_presence_in_both_lists():
    dense = hits("dense", [("a", 0.9), ("b", 0.8), ("c", 0.7)])
    sparse = hits("sparse", [("b", 11.0), ("d", 9.0)])
    fused = rrf([dense, sparse], k=60)
    order = [f["chunk_id"] for f in fused]
    assert order[0] == "b"                       # in both lists -> wins
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_scores_are_rank_based_not_score_based():
    # sparse scores 100x larger must NOT dominate: ranks only
    dense = hits("dense", [("a", 0.9), ("b", 0.8)])
    sparse = hits("sparse", [("b", 900.0), ("a", 850.0)])
    fused = rrf([dense, sparse], k=60)
    a = next(f for f in fused if f["chunk_id"] == "a")
    b = next(f for f in fused if f["chunk_id"] == "b")
    # a is rank1+rank2, b is rank2+rank1 -> identical fused score
    assert abs(a["fused_score"] - b["fused_score"]) < 1e-12


def test_rrf_records_source_ranks_for_debug():
    fused = rrf([hits("dense", [("a", 0.9)]), hits("sparse", [("a", 5.0)])])
    assert fused[0]["source_ranks"] == {"dense": 1, "sparse": 1}


def test_weighted_normalizes_before_mixing():
    dense = hits("dense", [("a", 0.9), ("b", 0.1)])
    sparse = hits("sparse", [("b", 500.0), ("a", 100.0)])
    fused = weighted([dense, sparse], [0.5, 0.5])
    scores = {f["chunk_id"]: f["fused_score"] for f in fused}
    assert abs(scores["a"] - scores["b"]) < 1e-9   # each wins one list fully
