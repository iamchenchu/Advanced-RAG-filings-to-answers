"""
db.py

The asyncpg connection pool, plus the tiny migration runner.

asyncpg speaks binary protocol and returns Records; there is no ORM here on
purpose. The queries in repository.py ARE the data-access documentation, and
SKIP LOCKED / jsonb / arrays are all clearer in SQL than through a mapper.
"""

import json
import pathlib

import asyncpg

from app.config import get_settings

_pool: asyncpg.Pool | None = None


async def _init_conn(conn: asyncpg.Connection) -> None:
    # jsonb <-> dict automatically, so repository code never json.dumps by hand
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = await asyncpg.create_pool(
            s.database_url,
            min_size=s.db_pool_min_size,
            max_size=s.db_pool_max_size,
            init=_init_conn,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def run_migrations(migrations_dir: str = "migrations") -> list[str]:
    """Apply every .sql file in order, once. Tracked in schema_migrations."""
    pool = await get_pool()
    applied: list[str] = []
    async with pool.acquire() as conn:
        # advisory lock: API + several workers all run migrations at startup;
        # without this, concurrent cold starts race the marker insert
        await conn.execute("SELECT pg_advisory_lock(742025)")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        done = {r["name"] for r in await conn.fetch("SELECT name FROM schema_migrations")}
        for path in sorted(pathlib.Path(migrations_dir).glob("*.sql")):
            if path.name in done:
                continue
            await conn.execute(path.read_text())
            await conn.execute(
                "INSERT INTO schema_migrations (name) VALUES ($1)", path.name
            )
            applied.append(path.name)
        await conn.execute("SELECT pg_advisory_unlock(742025)")
    return applied
