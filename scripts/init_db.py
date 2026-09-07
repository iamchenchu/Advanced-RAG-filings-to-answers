"""
scripts/init_db.py — apply migrations and create Qdrant collections.

    python scripts/init_db.py

Idempotent; the API also does both at startup. This exists so workers and
scripts can be run against a fresh database without booting the API first.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.qdrant import ensure_collections      # noqa: E402
from app.persistence.db import close_pool, run_migrations  # noqa: E402


async def main() -> None:
    applied = await run_migrations()
    print(f"migrations applied: {applied or 'none (up to date)'}")
    ensure_collections()
    print("qdrant collections ensured: chunks (HNSW) + chunks_flat (exact)")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
