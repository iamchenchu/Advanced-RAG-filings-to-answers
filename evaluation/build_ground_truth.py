"""
evaluation/build_ground_truth.py — verify AND repair the flat collection
against Postgres, the source of truth.

The old version compared only point counts, which can never remove a stray
point (review #38): after a failed deletion, count comparison sees
"flat has MORE than it should" and re-upserting fixes nothing. Now the check
is set-based: scroll every point id out of both collections, diff against
document_chunks, delete strays and re-queue missing ids.

    python evaluation/build_ground_truth.py
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.qdrant import ensure_collections, get_client   # noqa: E402
from app.config import get_settings                             # noqa: E402
from app.persistence.db import close_pool, get_pool             # noqa: E402


def all_point_ids(client, collection: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(collection_name=collection, limit=1000,
                                       offset=offset, with_payload=False,
                                       with_vectors=False)
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


async def main() -> None:
    s = get_settings()
    ensure_collections()
    client = get_client()
    pool = await get_pool()

    truth = {str(r["id"]) for r in await pool.fetch("SELECT id FROM document_chunks")}
    dirty = False
    for name in (s.qdrant_collection, s.qdrant_flat_collection):
        have = all_point_ids(client, name)
        stray = have - truth
        missing = truth - have
        print(f"{name}: {len(have)} points | {len(stray)} stray | {len(missing)} missing")
        if stray:
            from qdrant_client import models
            client.delete(collection_name=name,
                          points_selector=models.PointIdsList(points=sorted(stray)),
                          wait=True)
            print(f"  deleted {len(stray)} stray points")
        if missing:
            dirty = True
            await pool.executemany(
                "INSERT INTO vector_outbox (chunk_id) VALUES ($1) ON CONFLICT DO NOTHING",
                [(uuid.UUID(m),) for m in missing])
    if dirty:
        print("missing points re-queued; run: python workers/outbox_worker.py --once")
    else:
        print("ground truth is in sync with Postgres")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
