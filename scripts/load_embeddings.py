"""
scripts/load_embeddings.py — stage 3 of the Modal backfill: pull the
*.vectors.jsonl shards back from GCS, upsert chunk_vectors, enqueue outbox
rows. Fully idempotent — loading the same shard twice changes nothing.

    python scripts/load_embeddings.py --prefix embed_jobs/2026-08-18
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings                   # noqa: E402
from app.persistence.db import close_pool, get_pool   # noqa: E402


async def main(prefix: str) -> None:
    from google.cloud import storage

    s = get_settings()
    pool = await get_pool()
    client = storage.Client()
    bucket = client.bucket(s.gcs_bucket)

    total = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".vectors.jsonl"):
            continue
        rows = [json.loads(line) for line in
                blob.download_as_text().splitlines() if line.strip()]
        async with pool.acquire() as conn, conn.transaction():
            for r in rows:
                await conn.execute(
                    """UPDATE chunk_vectors SET dense = $2, model = $3
                       WHERE chunk_id = $1""",
                    uuid.UUID(r["chunk_id"]), r["dense"], r["model"])
                await conn.execute(
                    """UPDATE document_chunks SET embedding_model = $2 WHERE id = $1""",
                    uuid.UUID(r["chunk_id"]), r["model"])
                await conn.execute(
                    "INSERT INTO vector_outbox (chunk_id) VALUES ($1) ON CONFLICT DO NOTHING",
                    uuid.UUID(r["chunk_id"]))
        total += len(rows)
        print(f"  {blob.name}: {len(rows)} vectors loaded")
    print(f"{total} vectors -> chunk_vectors; now run: python workers/outbox_worker.py --once")
    await close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    asyncio.run(main(**vars(ap.parse_args())))
