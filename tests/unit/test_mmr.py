"""MMR — diversity must beat redundancy."""

from app.retrieval.mmr import mmr_select


def cand(cid, score, vec):
    return {"chunk_id": cid, "fused_score": score, "vector": vec, "text": cid}


def test_mmr_skips_near_duplicate_for_diverse_pick():
    a = cand("a", 1.00, [1.0, 0.0, 0.0])
    a2 = cand("a2", 0.99, [0.999, 0.01, 0.0])     # near-clone of a
    b = cand("b", 0.50, [0.0, 1.0, 0.0])          # different direction
    out = mmr_select([a, a2, b], top_n=2, lam=0.5)
    assert [c["chunk_id"] for c in out] == ["a", "b"]     # not a, a2


def test_lambda_one_is_pure_relevance():
    a = cand("a", 1.0, [1, 0, 0])
    a2 = cand("a2", 0.9, [1, 0, 0])
    b = cand("b", 0.1, [0, 1, 0])
    out = mmr_select([a, a2, b], top_n=2, lam=1.0)
    assert [c["chunk_id"] for c in out] == ["a", "a2"]    # duplicates fine at λ=1


def test_missing_vectors_pass_through():
    a = cand("a", 1.0, [1, 0])
    b = {"chunk_id": "b", "fused_score": 0.9, "vector": None, "text": "b"}
    out = mmr_select([a, b], top_n=2, lam=0.7)
    assert {c["chunk_id"] for c in out} == {"a", "b"}
