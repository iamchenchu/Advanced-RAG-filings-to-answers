"""
retrieval/fusion.py

Hand-written result fusion (PRD 3.5 requires exactly this — no server-side
fusion, because writing it is the point).

RRF — Reciprocal Rank Fusion:
    fused(d) = Σ_lists 1 / (k + rank_d)

Why ranks and not scores: dense cosine lives in [0, 1]-ish, BM25 lives in
[0, 20+]. Adding raw scores lets whichever list has bigger numbers win by
accident. Ranks are scale-free: RRF only asks "how high was this document in
each list", and k=60 (the constant from the original Cormack et al. paper)
damps the difference between rank 1 and rank 3 so one list cannot dominate.

weighted mode is the alternative for A/B: min-max normalize each list's
scores to [0,1], then weight-sum. More tunable, more fragile.
"""


def rrf(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Each dict needs 'chunk_id'. Returns merged, sorted, with fused_score
    and per-source ranks attached (the debug panel shows these)."""
    fused: dict[str, dict] = {}
    for source_idx, results in enumerate(result_lists):
        for rank, item in enumerate(results):
            cid = str(item["chunk_id"])
            entry = fused.setdefault(cid, {**item, "fused_score": 0.0, "source_ranks": {}})
            entry["fused_score"] += 1.0 / (k + rank + 1)
            entry["source_ranks"][item.get("source", f"list{source_idx}")] = rank + 1
            # keep the richest copy of shared fields (vectors, payload)
            for key, value in item.items():
                entry.setdefault(key, value)
    return sorted(fused.values(), key=lambda e: e["fused_score"], reverse=True)


def weighted(result_lists: list[list[dict]], weights: list[float]) -> list[dict]:
    fused: dict[str, dict] = {}
    for results, weight in zip(result_lists, weights):
        if not results:
            continue
        scores = [r["score"] for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        for item in results:
            cid = str(item["chunk_id"])
            entry = fused.setdefault(cid, {**item, "fused_score": 0.0, "source_ranks": {}})
            # all-equal scores (or a single result) must contribute full
            # weight, not collapse to 0 — being retrieved is the signal
            norm = (item["score"] - lo) / span if span else 1.0
            entry["fused_score"] += weight * norm
            for key, value in item.items():
                entry.setdefault(key, value)
    return sorted(fused.values(), key=lambda e: e["fused_score"], reverse=True)
