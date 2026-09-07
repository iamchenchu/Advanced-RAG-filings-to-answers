"""
scripts/make_shards_local.py — export every chunk still needing a BGE-M3
vector into JSONL shards under data/shards/, then print the exact
`modal volume put` command to ship them.

"Needing a vector" = dense IS NULL (the defer backend's rows) OR built by a
different model — the staleness query the embedding_model column exists for.

    python scripts/make_shards_local.py
    modal volume put rag-embed-shards data/shards /in
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings                    # noqa: E402
from app.persistence.db import close_pool, get_pool    # noqa: E402

OUT = Path("data/shards")
SHARD_SIZE = 2000


async def main() -> None:
    s = get_settings()
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT c.id, c.text, c.context_header, c.token_count
           FROM document_chunks c
           LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
           WHERE v.chunk_id IS NULL OR v.dense IS NULL
              OR v.model IS DISTINCT FROM $1
           ORDER BY c.document_id, c.chunk_index""",
        s.embedding_model,
    )
    print(f"{len(rows):,} chunks need {s.embedding_model} vectors")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("shard_*.jsonl"):
        old.unlink()                                   # fresh export every run

    for shard_no, start in enumerate(range(0, len(rows), SHARD_SIZE)):
        shard = rows[start:start + SHARD_SIZE]
        path = OUT / f"shard_{shard_no:05d}.jsonl"
        path.write_text("\n".join(json.dumps({
            "chunk_id": str(r["id"]),
            "embed_text": f"{r['context_header']}\n\n{r['text']}",
            "token_count": r["token_count"],
        }) for r in shard))
    n_shards = (len(rows) + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"{n_shards} shards -> {OUT}/")
    print("\nnext:")
    print("  modal volume put rag-embed-shards data/shards /in")
    print("  modal run modal_app/embed_h100.py")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
