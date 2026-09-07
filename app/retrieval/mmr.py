"""
retrieval/mmr.py

Maximal Marginal Relevance — hand-written (PRD 3.5).

The problem it solves: a 10-K repeats itself, and 64-token chunk overlap
guarantees near-duplicates. Pure relevance ranking happily fills all top-20
slots with five copies of the same paragraph. MMR trades a little relevance
for coverage:

    MMR(d) = λ * relevance(d) - (1 - λ) * max_{s ∈ selected} sim(d, s)

Greedy: pick the best, then repeatedly pick the candidate that is relevant
AND far from everything already picked. λ=1 is pure relevance, λ=0 is pure
diversity; 0.7 is the sane default.

sim() is cosine over the dense vectors that came back WITH the search results
(with_vectors=true), so MMR costs zero extra embedding calls.
"""

import numpy as np


def mmr_select(candidates: list[dict], top_n: int, lam: float) -> list[dict]:
    """candidates: fusion output, each ideally carrying 'vector' (list[float])
    and 'fused_score'. Items with no vector (sparse-only hits whose point was
    not fetched) are passed through by fused order at the end."""
    with_vec = [c for c in candidates if c.get("vector") is not None]
    without = [c for c in candidates if c.get("vector") is None]
    if not with_vec:
        return candidates[:top_n]

    vecs = np.asarray([c["vector"] for c in with_vec], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    sim = vecs @ vecs.T                                   # cosine, all pairs

    scores = np.asarray([c["fused_score"] for c in with_vec])
    lo, hi = scores.min(), scores.max()
    rel = (scores - lo) / ((hi - lo) or 1.0)              # normalize to [0,1]

    selected: list[int] = []
    remaining = set(range(len(with_vec)))
    while remaining and len(selected) < top_n:
        if not selected:
            best = int(np.argmax(rel))
        else:
            best, best_val = -1, -np.inf
            for i in remaining:
                penalty = max(sim[i][j] for j in selected)
                val = lam * rel[i] - (1 - lam) * penalty
                if val > best_val:
                    best, best_val = i, val
        selected.append(best)
        remaining.discard(best)

    out = [with_vec[i] for i in selected]
    for c in without:                                     # sparse-only tail
        if len(out) >= top_n:
            break
        out.append(c)
    return out
