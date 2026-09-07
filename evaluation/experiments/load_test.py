"""
Retrieval load test - concurrency vs latency (serving area, doc 03).

Fires N concurrent retrieval-only queries (generate:false) at the API and
plots how p50/p95/p99 and throughput move as concurrency climbs. Makes the
queue-vs-latency trade visible: past the point where every worker thread is
busy, added concurrency buys throughput at the cost of tail latency.

    python evaluation/experiments/load_test.py --url http://localhost:8000
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

QUERIES = [
    "supply chain risks", "revenue growth drivers", "cybersecurity threats",
    "executive compensation", "climate change risk", "goodwill impairment",
    "stock repurchase program", "litigation and legal proceedings",
    "research and development spend", "foreign currency exposure",
]
RESULTS = Path(__file__).resolve().parent.parent / "results" / "load_test.json"


async def one(client, url, q):
    t0 = time.perf_counter()
    r = await client.post(f"{url}/api/v1/query", json={"query": q, "top_k": 5, "generate": False})
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000


async def run_level(url, concurrency, total):
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)
        async def guarded(i):
            async with sem:
                return await one(client, url, QUERIES[i % len(QUERIES)])
        t0 = time.perf_counter()
        lat = await asyncio.gather(*[guarded(i) for i in range(total)])
        wall = time.perf_counter() - t0
    lat.sort()
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))]
    return {"concurrency": concurrency, "requests": total,
            "qps": round(total / wall, 1), "p50_ms": round(pct(0.50), 1),
            "p95_ms": round(pct(0.95), 1), "p99_ms": round(pct(0.99), 1)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--per-level", type=int, default=40)
    args = ap.parse_args()
    rows = []
    print(f"{'concurrency':>11} {'qps':>7} {'p50':>7} {'p95':>7} {'p99':>7}")
    print("-" * 44)
    for c in (1, 2, 4, 8, 16):
        row = await run_level(args.url, c, args.per_level)
        rows.append(row)
        print(f"{row['concurrency']:>11} {row['qps']:>7} {row['p50_ms']:>7} "
              f"{row['p95_ms']:>7} {row['p99_ms']:>7}")
    RESULTS.write_text(json.dumps({"levels": rows}, indent=2))
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
