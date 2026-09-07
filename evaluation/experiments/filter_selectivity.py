"""
Experiment 5 - filter selectivity vs recall (PRD 7.4).

The question: as a metadata pre-filter gets more selective (matches fewer and
fewer points), does HNSW still return a full, correct top-K? Post-filtering
would silently return too few results; TRUE pre-filtering (Qdrant applies the
filter during graph traversal) should hold recall and full K.

For each selectivity level we pick a filter that matches a known fraction of
the 1.73M-point corpus, then compare the HNSW result against an EXACT filtered
search (the flat collection with the same filter) - recall is defined against
that, exactly as in run_eval.py.

    python evaluation/experiments/filter_selectivity.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.clients.embedder import get_embedder          # noqa: E402
from app.config import get_settings                    # noqa: E402
from app.persistence.db import close_pool, get_pool    # noqa: E402
from app.retrieval.dense import dense_search           # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "filter_selectivity.json"
QUERY = "supply chain and manufacturing risks"
K = 10


async def main() -> None:
    s = get_settings()
    pool = await get_pool()
    total = await pool.fetchval("SELECT count(*) FROM document_chunks")

    # build filters of decreasing selectivity, measuring the real match fraction
    async def frac(where: str, *args) -> float:
        n = await pool.fetchval(
            f"SELECT count(*) FROM document_chunks c "
            f"JOIN documents d ON d.id=c.document_id WHERE {where}", *args)
        return n / total

    levels = [
        ("no filter", None,
         1.0),
        ("sector-ish: 50 tickers", {"tickers_stub": True},
         await frac("d.meta->>'ticker' = ANY($1)",
                    [r["t"] for r in await pool.fetch(
                        "SELECT DISTINCT d.meta->>'ticker' t FROM documents d LIMIT 50")])),
        ("one ticker (AAPL)", {"ticker": "AAPL"},
         await frac("d.meta->>'ticker' = 'AAPL'")),
        ("one ticker + one item", {"ticker": "AAPL", "section": "1A"},
         await frac("d.meta->>'ticker'='AAPL' AND c.section='1A'")),
        ("one ticker + item + year", {"ticker": "AAPL", "section": "1A", "fiscal_year": "2024"},
         await frac("d.meta->>'ticker'='AAPL' AND c.section='1A' AND d.meta->>'fiscal_year'='2024'")),
    ]

    vec = get_embedder().embed_one(QUERY).tolist()
    rows = []
    print(f"{'selectivity level':<34} {'match %':>8} {'recall@10':>10} {'p50 ms':>8} {'returned':>9}")
    print("-" * 74)
    for name, filt, fraction in levels:
        f = None if filt is None else {k: v for k, v in filt.items() if k != "tickers_stub"}
        # tickers_stub level needs a real 'tickers' filter the API doesn't model; skip its ANN and just report fraction
        if filt and filt.get("tickers_stub"):
            print(f"{name:<34} {fraction*100:>7.2f}% {'(informational - multi-ticker filter not in API)':>10}")
            rows.append({"level": name, "match_pct": round(fraction*100, 3)})
            continue
        truth = dense_search(vec, K, collection=s.qdrant_flat_collection, exact=True, filters=f)
        truth_ids = [str(h["chunk_id"]) for h in truth]
        t0 = time.perf_counter()
        got = dense_search(vec, K, filters=f)
        dt = (time.perf_counter() - t0) * 1000
        got_ids = [str(h["chunk_id"]) for h in got]
        recall = (len(set(got_ids) & set(truth_ids)) / len(truth_ids)) if truth_ids else 1.0
        print(f"{name:<34} {fraction*100:>7.3f}% {recall:>10.3f} {dt:>8.1f} {len(got_ids):>9}")
        rows.append({"level": name, "match_pct": round(fraction*100, 3),
                     "recall_at_10": round(recall, 3), "p50_ms": round(dt, 1),
                     "returned": len(got_ids), "truth_available": len(truth_ids)})

    RESULTS.write_text(json.dumps({"query": QUERY, "corpus_points": total, "levels": rows}, indent=2))
    print(f"\nwrote {RESULTS}")
    print("reading: if recall stays ~1.0 and 'returned' stays = K as match% shrinks,")
    print("pre-filtering is working - the filter is applied DURING traversal, not after.")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
