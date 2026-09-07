"""
evaluation/experiments/ef_sweep.py — EXPERIMENT 1 (PRD 7.4: do this first).

Sweeps ef_search — the query-time size of the HNSW candidate list — and
records recall vs latency at each point. No rebuilds: ef_search rides on the
query. The output curve answers the only question that matters for tuning:
"what does one more point of recall cost in milliseconds, on MY data?"

    python evaluation/experiments/ef_sweep.py
    python evaluation/experiments/ef_sweep.py --ef 16 32 64 128 256 512
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.persistence.db import close_pool          # noqa: E402
from evaluation.run_eval import evaluate_recall    # noqa: E402

RESULTS = Path(__file__).parent.parent / "results" / "ef_sweep.json"


async def main(ef_values: list[int]) -> None:
    rows = []
    for ef in ef_values:
        summary = await evaluate_recall(config_name=f"ef{ef}", ef_search=ef,
                                        k_values=[1, 5, 10])
        rows.append({"ef_search": ef,
                     "recall@10": summary["recall@10"],
                     "latency_p50_ms": summary["latency_p50_ms"],
                     "latency_p95_ms": summary["latency_p95_ms"]})

    RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"\n{'ef':>5} {'recall@10':>10} {'p50 ms':>8} {'p95 ms':>8}")
    for r in rows:
        print(f"{r['ef_search']:>5} {r['recall@10']:>10.4f} "
              f"{r['latency_p50_ms']:>8.2f} {r['latency_p95_ms']:>8.2f}")
    print(f"\nwrote {RESULTS}")
    await close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    args = ap.parse_args()
    asyncio.run(main(args.ef))
