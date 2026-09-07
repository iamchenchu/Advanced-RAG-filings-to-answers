"""
scripts/load_vectors.py — final backfill stage: H100 vectors -> Postgres ->
outbox. Idempotent; re-running re-applies the same updates harmlessly.

    modal volume get rag-embed-shards /out data/vectors
    python scripts/load_vectors.py
    python workers/outbox_worker.py            # -> Qdrant, both collections
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings                    # noqa: E402
from app.persistence.db import close_pool, get_pool    # noqa: E402

VEC_DIR = Path("data/vectors")


async def main() -> None:
    s = get_settings()
    pool = await get_pool()
    total = missing = 0
    shards = sorted(VEC_DIR.glob("shard_*.ids.json"))
    if not shards:
        print(f"no vector shards under {VEC_DIR}/ — run `modal volume get` first")
        return

    for ids_path in shards:
        npy_path = ids_path.with_name(ids_path.name.replace(".ids.json", ".f16.npy"))
        ids = json.loads(ids_path.read_text())
        matrix = np.load(npy_path).astype(np.float32)
        assert len(ids) == matrix.shape[0], f"{ids_path.name}: ids/matrix mismatch"

        args = [(uuid.UUID(cid), [float(x) for x in vec], s.embedding_model)
                for cid, vec in zip(ids, matrix)]
        async with pool.acquire() as conn, conn.transaction():
            result = await conn.executemany(
                """UPDATE chunk_vectors SET dense = $2, model = $3
                   WHERE chunk_id = $1""", args)
            await conn.executemany(
                "INSERT INTO vector_outbox (chunk_id) VALUES ($1) ON CONFLICT DO NOTHING",
                [(a[0],) for a in args])
        total += len(ids)
        print(f"{ids_path.stem.removesuffix('.ids')}: {len(ids)} vectors loaded "
              f"({total:,} total)")

    still = await pool.fetchval(
        "SELECT count(*) FROM chunk_vectors WHERE dense IS NULL")
    print(f"\n{total:,} vectors loaded | chunks still without vectors: {still}")
    print("next: python workers/outbox_worker.py   (drains outbox -> Qdrant)")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
