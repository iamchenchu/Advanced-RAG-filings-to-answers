"""
workers/outbox_worker.py

Drains vector_outbox into Qdrant. The second half of the transactional-outbox
pattern: Postgres commit made the chunk durable; this worker makes it
SEARCHABLE, and the gap between the two is visible as rag_outbox_pending_total.

BM25 index-time weights are computed HERE, at sync time, with the corpus
statistics as they are NOW — so a chunk synced today is weighted against
today's avgdl. (After a large corpus change, `--resync` re-queues everything
so old points get re-weighted too.)

Idempotent by construction: point id == chunk id, upsert overwrites. Sync the
same row twice and Qdrant holds one point.

Run:  python workers/outbox_worker.py             # poll forever
      python workers/outbox_worker.py --once      # drain, then exit
      python workers/outbox_worker.py --resync    # re-queue ALL chunks first
"""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.qdrant import (delete_points, ensure_collections,  # noqa: E402
                                upsert_points)
from app.config import get_settings                                # noqa: E402
from app.persistence import repository                             # noqa: E402
from app.persistence.db import close_pool, run_migrations          # noqa: E402
from app.retrieval.sparse import index_weights                     # noqa: E402


async def drain(once: bool) -> None:
    s = get_settings()
    await run_migrations()
    ensure_collections()
    max_attempts = 10
    # a previous worker that died mid-batch leaves rows in 'syncing'
    reset = await repository.reset_stale_syncing()
    if reset:
        print(f"reset {reset} stale syncing row(s) from a dead worker")
    print(f"outbox worker up -> {s.qdrant_url} "
          f"({s.qdrant_collection} + {s.qdrant_flat_collection})")

    while True:
        rows = await repository.claim_pending_outbox(s.outbox_batch_size, max_attempts)
        did_work = bool(rows)
        if rows:
            stats = await repository.corpus_stats()
            weights = [
                index_weights({int(k): v for k, v in (r["sparse_tf"] or {}).items()},
                              r["doc_len"] or 0, stats["avgdl"], s.bm25_k1, s.bm25_b)
                for r in rows
            ]
            try:
                upsert_points(rows, weights)
                await repository.mark_outbox([r["outbox_id"] for r in rows], "synced")
                print(f"synced {len(rows)} points (avgdl={stats['avgdl']:.1f})")
            except Exception:                       # noqa: BLE001
                # batch failed: isolate the poison row(s) by retrying one by
                # one, so one bad point cannot head-of-line block the rest
                traceback.print_exc()
                ok = bad = 0
                for row, weight in zip(rows, weights):
                    try:
                        upsert_points([row], [weight])
                        await repository.mark_outbox([row["outbox_id"]], "synced")
                        ok += 1
                    except Exception as exc:        # noqa: BLE001
                        await repository.mark_outbox(
                            [row["outbox_id"]], "pending",
                            f"{type(exc).__name__}: {exc}", max_attempts)
                        bad += 1
                print(f"row-isolation pass: {ok} synced, {bad} re-queued/dead-lettered")
                await asyncio.sleep(2)

        # tombstones AFTER upserts: an in-flight upsert claimed before a
        # delete commits gets purged by the tombstone in this same cycle,
        # so a deleted document can never permanently resurrect (review #3)
        dels = await repository.claim_pending_deletes(s.outbox_batch_size, max_attempts)
        if dels:
            did_work = True
            try:
                delete_points([d["chunk_id"] for d in dels])
                await repository.mark_deletes([d["id"] for d in dels], "synced")
                print(f"purged {len(dels)} deleted points from Qdrant")
            except Exception as exc:                # noqa: BLE001
                await repository.mark_deletes([d["id"] for d in dels], "pending",
                                              max_attempts)
                print(f"tombstone purge failed, re-queued: {exc}")
                await asyncio.sleep(2)

        if not did_work:
            if once:
                print("outbox empty, exiting")
                return
            await asyncio.sleep(s.outbox_poll_interval_s)


def _maybe_serve_metrics() -> None:
    """Workers mutate their own metric registry (INGESTION_FAILED etc), which
    the API's /metrics can never see (separate process). Set METRICS_PORT to
    expose this worker's registry for Prometheus to scrape."""
    import os
    port = os.getenv("METRICS_PORT")
    if port:
        from prometheus_client import start_http_server
        start_http_server(int(port))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--resync", action="store_true",
                    help="queue every chunk for re-sync (rebuild collections from Postgres)")
    args = ap.parse_args()
    _maybe_serve_metrics()

    async def run() -> None:
        try:
            if args.resync:
                n = await repository.requeue_all_chunks()
                print(f"re-queued {n} chunks for full re-sync")
            await drain(args.once)
        finally:
            await close_pool()      # same event loop as the pool — clean exit

    asyncio.run(run())


if __name__ == "__main__":
    main()
