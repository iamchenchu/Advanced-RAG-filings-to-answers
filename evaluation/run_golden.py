"""
evaluation/run_golden.py — the LABELED retrieval evaluation, per config.

For every golden query, runs four retrieval configurations and scores each
against the labeled relevant chunks:

    dense    bi-encoder vectors only
    sparse   hand-written BM25 only
    hybrid   dense + sparse -> RRF fusion
    full     hybrid -> MMR -> cross-encoder rerank   (the production path)

Metrics: hit@1/5/10 (a relevant chunk appears in top K) and MRR (rank of the
first relevant chunk). This is the PRD's "measured delta per stage" table —
the number that justifies (or indicts) each pipeline stage.

No ticker filters are passed: the pipeline must find Apple's answer inside
the full multi-company corpus, which is the honest setting.

    python evaluation/run_golden.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.embedder import get_embedder          # noqa: E402
from app.clients.reranker import RerankUnavailable, get_reranker  # noqa: E402
from app.config import get_settings                    # noqa: E402
from app.persistence.db import close_pool              # noqa: E402
from app.retrieval.dense import dense_search           # noqa: E402
from app.retrieval.fusion import rrf                   # noqa: E402
from app.retrieval.mmr import mmr_select               # noqa: E402
from app.retrieval.sparse_search import sparse_search  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "golden.jsonl"
RESULTS = Path(__file__).parent / "results" / "golden_eval.json"
K_VALUES = (1, 5, 10)
TOP_K = 10


def score(ranked_ids: list[str], relevant: set[str]) -> dict:
    out = {}
    for k in K_VALUES:
        out[f"hit@{k}"] = float(any(cid in relevant for cid in ranked_ids[:k]))
    rr = 0.0
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


async def run_configs(entry: dict, pool_db) -> dict[str, dict]:
    import uuid as _uuid
    s = get_settings()
    query = entry["query"]
    relevant = set(entry["relevant_chunk_ids"])

    vec = get_embedder().embed_one(query).tolist()
    dense_hits = dense_search(vec, s.top_k_dense)
    sparse_hits = await sparse_search(query, s.top_k_sparse)
    fused = rrf([dense_hits, sparse_hits], k=s.rrf_k)

    # full production path: MMR then cross-encoder over the pool.
    # hydrate ONLY the pool's texts — hydrating the whole corpus was fine at
    # 2k chunks and is ~4GB of RAM at 1.7M
    pool = mmr_select(fused, s.mmr_top_n, s.mmr_lambda)
    ids = [_uuid.UUID(str(h["chunk_id"])) for h in pool]
    rows = await pool_db.fetch(
        "SELECT id, text FROM document_chunks WHERE id = ANY($1)", ids)
    texts_by_id = {str(r["id"]): r["text"] for r in rows}
    for h in pool:
        h["text"] = texts_by_id.get(str(h["chunk_id"]), "")
    try:
        full = get_reranker().rerank(query, [h for h in pool if h["text"]])
    except RerankUnavailable:
        full = pool

    return {
        "dense":  score([str(h["chunk_id"]) for h in dense_hits], relevant),
        "sparse": score([str(h["chunk_id"]) for h in sparse_hits], relevant),
        "hybrid": score([str(h["chunk_id"]) for h in fused], relevant),
        "full":   score([str(h["chunk_id"]) for h in full], relevant),
    }


async def main() -> None:
    from app.persistence.db import get_pool
    entries = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    pool_db = await get_pool()
    print(f"{len(entries)} golden queries")

    agg: dict[str, dict[str, list[float]]] = {}
    per_query = []
    for entry in entries:
        configs = await run_configs(entry, pool_db)
        per_query.append({"query": entry["query"], **{c: m for c, m in configs.items()}})
        for config, metrics in configs.items():
            for metric, value in metrics.items():
                agg.setdefault(config, {}).setdefault(metric, []).append(value)

    print(f"\n{'config':<8} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'mrr':>7}")
    print("-" * 42)
    summary = {}
    for config in ("dense", "sparse", "hybrid", "full"):
        m = {metric: round(sum(v) / len(v), 4) for metric, v in agg[config].items()}
        summary[config] = m
        print(f"{config:<8} {m['hit@1']:>7.3f} {m['hit@5']:>7.3f} "
              f"{m['hit@10']:>7.3f} {m['mrr']:>7.3f}")

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))
    print(f"\nwrote {RESULTS}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
