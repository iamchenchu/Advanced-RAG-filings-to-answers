"""
evaluation/run_eval.py — Recall@K, MRR and latency of the HNSW index,
measured against the flat (exact) collection. PRD 7.3: Recall@K is UNDEFINED
without the flat ground truth; this is why chunks_flat exists.

For every query:
  truth   = exact search on chunks_flat  (what perfect retrieval returns)
  result  = HNSW search on chunks        (what production actually returns)
  recall@k = |result_k ∩ truth_k| / k

No labeled relevance judgments are needed for THIS metric: it isolates the
ANN index's approximation error, not embedding quality. (Whether the embedding
model retrieves relevant text is a separate, human-labeled question.)

NOTE at small corpus size: below HNSW_FULL_SCAN_THRESHOLD (10k) Qdrant
brute-forces even the HNSW collection, so recall == 1.0 by construction.
The harness still proves the loop; the numbers become interesting at scale.

    python evaluation/run_eval.py
    python evaluation/run_eval.py --config ef64 --ef-search 64 --k 1 5 10
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.embedder import get_embedder            # noqa: E402
from app.config import get_settings                      # noqa: E402
from app.observability.metrics import RECALL_AT_10       # noqa: E402
from app.persistence import repository                   # noqa: E402
from app.persistence.db import close_pool                # noqa: E402
from app.retrieval.dense import dense_search             # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "queries.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_queries() -> list[str]:
    return [json.loads(line)["query"]
            for line in FIXTURES.read_text().splitlines() if line.strip()]


def _pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile via ceil — the old int(n*q)-1 floored the rank
    and systematically under-reported p95 (review #43)."""
    import math
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1)]


async def evaluate_recall(run_id: uuid.UUID | None = None,
                          k_values: list[int] | None = None,
                          ef_search: int | None = None,
                          config_name: str = "default") -> dict:
    s = get_settings()
    # k <= 0 would divide by zero / produce negative "recall" (review #42)
    k_values = sorted(k for k in (k_values or [1, 5, 10]) if k >= 1) or [10]
    k_max = max(k_values)
    try:
        queries = load_queries()
        embedder = get_embedder()
    except Exception as exc:                        # noqa: BLE001
        # a run row stuck 'running' forever tells the 202 caller nothing
        # (review #39) — record the failure, then re-raise
        if run_id is not None:
            await repository.finish_eval_run(
                run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise

    per_query: list[tuple[str, dict]] = []
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: list[float] = []
    latencies: list[float] = []

    try:
      for query in queries:
        vec = embedder.embed_one(query).tolist()

        truth = dense_search(vec, k_max, collection=s.qdrant_flat_collection, exact=True)
        truth_ids = [str(h["chunk_id"]) for h in truth]

        t0 = time.perf_counter()
        result = dense_search(vec, k_max, ef_search=ef_search)
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids = [str(h["chunk_id"]) for h in result]

        metrics: dict = {}
        for k in k_values:
            hits = len(set(result_ids[:k]) & set(truth_ids[:k]))
            recall = hits / min(k, len(truth_ids)) if truth_ids else 0.0
            metrics[f"recall@{k}"] = round(recall, 4)
            recalls[k].append(recall)
        # MRR: rank of the true best chunk inside the HNSW result
        rr = 0.0
        if truth_ids:
            for rank, cid in enumerate(result_ids, start=1):
                if cid == truth_ids[0]:
                    rr = 1.0 / rank
                    break
        metrics["mrr"] = round(rr, 4)
        mrrs.append(rr)
        per_query.append((query, metrics))
    except Exception as exc:                        # noqa: BLE001
        if run_id is not None:
            await repository.finish_eval_run(
                run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
        raise

    lat_sorted = sorted(latencies)
    summary = {
        "config_name": config_name,
        "ef_search": ef_search or s.hnsw_ef_search,
        "n_queries": len(queries),
        **{f"recall@{k}": round(statistics.mean(v), 4) for k, v in recalls.items()},
        "mrr": round(statistics.mean(mrrs), 4),
        "latency_p50_ms": round(_pct(lat_sorted, 0.50), 2),
        "latency_p95_ms": round(_pct(lat_sorted, 0.95), 2),
        "latency_p99_ms": round(_pct(lat_sorted, 0.99), 2),
        "embedding_model": s.embedding_model,
        "hnsw": {"m": s.hnsw_m, "ef_construct": s.hnsw_ef_construct},
        "quantization": s.quantization,
    }
    if 10 in recalls:
        RECALL_AT_10.set(summary["recall@10"])

    if run_id is not None:
        await repository.add_eval_results(run_id, per_query)
        await repository.finish_eval_run(run_id, "done", summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"recall_{config_name}.json"
    out.write_text(json.dumps({"summary": summary,
                               "per_query": [{"query": q, **m} for q, m in per_query]},
                              indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--ef-search", type=int, default=None)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()

    async def run() -> None:
        run_id = await repository.create_eval_run(
            args.config, {"ef_search": args.ef_search, "k": args.k})
        await evaluate_recall(run_id, args.k, args.ef_search, args.config)
        await close_pool()

    asyncio.run(run())
