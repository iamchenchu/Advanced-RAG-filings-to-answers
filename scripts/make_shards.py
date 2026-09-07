"""
scripts/make_shards.py — stage 1 of the Modal backfill: export every chunk
that still needs a vector from THIS embedding model into GCS JSONL shards.

    python scripts/make_shards.py --model BAAI/bge-m3 --prefix embed_jobs/2026-08-18
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings          # noqa: E402
from app.persistence.db import close_pool, get_pool  # noqa: E402

SHARD_SIZE = 2000       # ~2k chunks x ~2.5KB embed_text ≈ 5MB per shard


async def main(model: str, prefix: str) -> None:
    from google.cloud import storage

    s = get_settings()
    pool = await get_pool()
    # chunks whose vector is missing OR was built by a different model —
    # exactly the "find stale vectors" query the embedding_model column exists for
    rows = await pool.fetch(
        """SELECT c.id, c.text, c.context_header, c.token_count
           FROM document_chunks c
           LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
           WHERE v.chunk_id IS NULL OR v.dense IS NULL OR v.model IS DISTINCT FROM $1
           ORDER BY c.document_id, c.chunk_index""",
        model,
    )
    print(f"{len(rows)} chunks need embeddings from {model}")

    client = storage.Client()
    bucket = client.bucket(s.gcs_bucket)
    for shard_no, start in enumerate(range(0, len(rows), SHARD_SIZE)):
        shard = rows[start : start + SHARD_SIZE]
        payload = "\n".join(json.dumps({
            "chunk_id": str(r["id"]),
            "embed_text": f"{r['context_header']}\n\n{r['text']}",
            "token_count": r["token_count"],
        }) for r in shard)
        name = f"{prefix}/shard_{shard_no:05d}.jsonl"
        bucket.blob(name).upload_from_string(payload)
        print(f"  gs://{s.gcs_bucket}/{name}  ({len(shard)} chunks)")
    await close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--prefix", required=True)
    asyncio.run(main(**vars(ap.parse_args())))
