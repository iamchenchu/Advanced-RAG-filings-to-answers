"""Context builder — budget, caps, dedupe, citation integrity."""

from app.config import get_settings
from app.retrieval.context_builder import build_context


def cand(cid, doc="d1", tokens=100, text=None, vec=None, score=1.0):
    return {"chunk_id": cid, "document_id": doc, "document_name": f"{doc}.pdf",
            "section": "1A", "section_title": "Risk Factors",
            "token_count": tokens, "text": text or f"text of {cid}",
            "vector": vec, "fused_score": score}


def test_citation_ids_sequential_and_resolve():
    ctx = build_context([cand("a"), cand("b", doc="d2"), cand("c", doc="d3")])
    assert [c["id"] for c in ctx["citations"]] == [1, 2, 3]
    for c in ctx["citations"]:
        assert f"[{c['id']}]" in ctx["context_text"]      # id appears in prompt
        assert c["chunk_id"] and c["text"]                # resolves to a real chunk


def test_token_budget_is_hard():
    s = get_settings()
    budget = s.max_context_tokens - s.context_reserve_tokens
    huge = [cand(f"c{i}", doc=f"d{i}", tokens=budget // 2 + 1) for i in range(5)]
    ctx = build_context(huge)
    assert ctx["context_tokens"] <= budget
    assert len(ctx["citations"]) == 1                     # only one fits
    assert ctx["dropped"]["budget"] == 4


def test_per_document_cap():
    s = get_settings()
    many_same_doc = [cand(f"c{i}", doc="d1") for i in range(10)]
    ctx = build_context(many_same_doc)
    assert len(ctx["citations"]) == s.max_chunks_per_document
    assert ctx["dropped"]["doc_cap"] == 10 - s.max_chunks_per_document


def test_near_duplicates_suppressed():
    a = cand("a", vec=[1.0, 0.0])
    dup = cand("dup", doc="d2", vec=[0.999, 0.02])        # cosine ~0.999
    diff = cand("diff", doc="d3", vec=[0.0, 1.0])
    ctx = build_context([a, dup, diff])
    ids = [c["chunk_id"] for c in ctx["citations"]]
    assert "a" in ids and "diff" in ids and "dup" not in ids
    assert ctx["dropped"]["duplicate"] == 1
