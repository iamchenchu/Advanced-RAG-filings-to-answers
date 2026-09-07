"""
repository.py

Every SQL statement in the system lives here. Handlers and workers call these
functions; nothing else talks to rag_db.

The two patterns worth reading closely:

  claim_job        — FOR UPDATE SKIP LOCKED. N workers can poll the same table
                     and never hand out the same job twice, with zero extra
                     infrastructure. This is "a queue" in 12 lines of SQL.

  insert_chunks    — chunks, vectors, BM25 stats, and outbox rows commit in
                     ONE transaction. Either a chunk exists WITH its stats and
                     its pending-sync marker, or none of it exists. That
                     atomicity is what makes worker death harmless.
"""

import uuid
from typing import Any

import asyncpg

from app.persistence.db import get_pool

# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

async def create_document(
    name: str, source_type: str, gcs_uri: str | None,
    tenant: str, region: str, tags: list, meta: dict,
) -> dict:
    """Insert a document and its ingestion job atomically; both or neither.

    Re-registering the SAME gcs_uri reuses the existing document row and just
    enqueues a fresh job (idempotent chunking makes the re-run a no-op unless
    something actually changed). Without this, every retry of a backfill
    script would mint a duplicate ready-but-empty document row."""
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        doc = None
        if gcs_uri:
            # reuse AND refresh: silently discarding new name/meta/tags on a
            # re-registration made metadata corrections impossible (review #33)
            doc = await conn.fetchrow(
                """UPDATE documents SET name = $3, source_type = $4, region = $5,
                       tags = $6, meta = $7, updated_at = now()
                   WHERE gcs_uri = $1 AND tenant = $2 RETURNING *""",
                gcs_uri, tenant, name, source_type, region,
                list(tags or []), dict(meta or {}))
        if doc is None:
            doc = await conn.fetchrow(
                """INSERT INTO documents (name, source_type, gcs_uri, tenant, region, tags, meta)
                   VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *""",
                name, source_type, gcs_uri, tenant, region, list(tags or []), dict(meta or {}),
            )
        job = await conn.fetchrow(
            "INSERT INTO ingestion_jobs (document_id) VALUES ($1) RETURNING id",
            doc["id"],
        )
    return {**dict(doc), "job_id": job["id"]}


async def list_documents(tenant: str | None = None, limit: int = 100) -> list[dict]:
    pool = await get_pool()
    if tenant:
        rows = await pool.fetch(
            """SELECT d.*, (SELECT count(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunk_count
               FROM documents d WHERE tenant = $1 ORDER BY created_at DESC LIMIT $2""",
            tenant, limit)
    else:
        rows = await pool.fetch(
            """SELECT d.*, (SELECT count(*) FROM document_chunks c WHERE c.document_id = d.id) AS chunk_count
               FROM documents d ORDER BY created_at DESC LIMIT $1""", limit)
    return [dict(r) for r in rows]


async def get_document(document_id: uuid.UUID) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM documents WHERE id = $1", document_id)
    return dict(row) if row else None


async def document_status(document_id: uuid.UUID) -> dict | None:
    """Everything the /status endpoint needs in one round trip."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT d.id, d.name, d.status, d.error, d.created_at, d.updated_at,
                  (SELECT count(*) FROM document_chunks c WHERE c.document_id = d.id)  AS chunks,
                  (SELECT count(*) FROM vector_outbox o
                     JOIN document_chunks c ON c.id = o.chunk_id
                    WHERE c.document_id = d.id AND o.status = 'pending')               AS pending_sync,
                  (SELECT jsonb_agg(jsonb_build_object(
                        'id', j.id, 'status', j.status, 'attempts', j.attempts,
                        'last_error', j.last_error, 'started_at', j.started_at,
                        'finished_at', j.finished_at) ORDER BY j.id)
                     FROM ingestion_jobs j WHERE j.document_id = d.id)                 AS jobs
           FROM documents d WHERE d.id = $1""",
        document_id,
    )
    return dict(row) if row else None


async def set_document_status(document_id: uuid.UUID, status: str, error: str | None = None) -> None:
    pool = await get_pool()
    await pool.execute(
        "UPDATE documents SET status = $2, error = $3, updated_at = now() WHERE id = $1",
        document_id, status, error,
    )


async def delete_document(document_id: uuid.UUID) -> list[uuid.UUID]:
    """Delete a document. Returns the removed chunk ids (for the response).

    Everything happens in ONE transaction:
      - FOR UPDATE on the chunk rows serializes concurrent deletes of the same
        document, so BM25 stats cannot be double-decremented (review #9)
      - stats are decremented alongside the rows they describe
      - Qdrant purge is recorded as TOMBSTONES in vector_deletes and drained
        by the outbox worker — crash-safe, retryable, no orphan points
        (reviews #2/#20/#35)
    """
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        chunk_ids = [r["id"] for r in await conn.fetch(
            "SELECT id FROM document_chunks WHERE document_id = $1 FOR UPDATE",
            document_id)]
        if chunk_ids:
            # decrement df for every distinct term of every chunk being removed
            await conn.execute(
                """WITH terms AS (
                       SELECT (jsonb_each_text(v.sparse_tf)).key::bigint AS term_id
                       FROM chunk_vectors v WHERE v.chunk_id = ANY($1)
                   ), counted AS (
                       SELECT term_id, count(*) AS n FROM terms GROUP BY term_id
                   )
                   UPDATE bm25_stats s SET df = greatest(0, s.df - counted.n)
                   FROM counted WHERE s.term_id = counted.term_id""",
                chunk_ids,
            )
            stats = await conn.fetchrow(
                """SELECT count(*) AS n, coalesce(sum(doc_len), 0) AS len
                   FROM chunk_vectors WHERE chunk_id = ANY($1)""", chunk_ids)
            await conn.execute(
                "UPDATE bm25_corpus SET n_chunks = greatest(0, n_chunks - $1),"
                " total_len = greatest(0, total_len - $2) WHERE id = 1",
                stats["n"], stats["len"])
            await conn.executemany(
                "INSERT INTO vector_deletes (chunk_id) VALUES ($1)",
                [(c,) for c in chunk_ids])
        await conn.execute("DELETE FROM documents WHERE id = $1", document_id)  # cascades
    return chunk_ids


# ---------------------------------------------------------------------------
# ingestion jobs — the SKIP LOCKED queue
# ---------------------------------------------------------------------------

async def claim_job(worker_id: str, max_attempts: int) -> dict | None:
    """Atomically claim one pending job. Concurrent workers never collide:
    FOR UPDATE locks the row, SKIP LOCKED makes competitors skip it instead
    of waiting, LIMIT 1 takes the oldest."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """UPDATE ingestion_jobs SET
               status = 'processing', worker_id = $1,
               started_at = now(), attempts = attempts + 1
           WHERE id = (
               SELECT id FROM ingestion_jobs
               WHERE status = 'pending' AND attempts < $2
               ORDER BY id
               FOR UPDATE SKIP LOCKED
               LIMIT 1)
           RETURNING *""",
        worker_id, max_attempts,
    )
    return dict(row) if row else None


async def finish_job(job_id: int, status: str, error: str | None = None,
                     worker_id: str | None = None) -> bool:
    """Finish a job. With worker_id set, the update is FENCED: it only lands
    if this worker still owns the job (review #5 — a reaped job whose original
    worker wakes up late must not clobber the reassigned run). Returns False
    when the fence rejected the update."""
    pool = await get_pool()
    if worker_id is not None:
        result = await pool.execute(
            """UPDATE ingestion_jobs SET status = $2, last_error = $3, finished_at = now()
               WHERE id = $1 AND worker_id = $4 AND status = 'processing'""",
            job_id, status, error, worker_id)
    else:
        result = await pool.execute(
            """UPDATE ingestion_jobs SET status = $2, last_error = $3, finished_at = now()
               WHERE id = $1""",
            job_id, status, error)
    return result.endswith("1")


async def reap_stuck_jobs(timeout_minutes: int, max_attempts: int) -> int:
    """Jobs stuck in 'processing' past the timeout: the worker died mid-job.
    Under max_attempts they go back to pending (the whole pipeline is
    idempotent, so re-running is safe); at max_attempts they are dead-lettered
    for a human, never retried forever."""
    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        reset = await conn.fetch(
            """UPDATE ingestion_jobs SET status = CASE
                     WHEN attempts >= $2 THEN 'dead' ELSE 'pending' END,
                   last_error = coalesce(last_error, '') || ' [reaped: worker timeout]'
               WHERE status = 'processing'
                 AND started_at < now() - make_interval(mins => $1)
               RETURNING id, document_id, status""",
            timeout_minutes, max_attempts,
        )
        # a dead-lettered job must not leave its document 'processing' forever
        dead_docs = [r["document_id"] for r in reset if r["status"] == "dead"]
        if dead_docs:
            await conn.execute(
                """UPDATE documents SET status = 'failed',
                       error = 'ingestion dead-lettered (worker timeout)',
                       updated_at = now()
                   WHERE id = ANY($1)""", dead_docs)
    return len(reset)


# ---------------------------------------------------------------------------
# chunks + vectors + stats + outbox — the ONE-transaction insert
# ---------------------------------------------------------------------------

async def insert_chunks(document_id: uuid.UUID, chunks: list[dict]) -> dict:
    """RECONCILE the document's chunks to exactly `chunks`, in one transaction.

    This replaced a naive ON CONFLICT DO NOTHING insert after the adversarial
    review proved three failure modes (findings #1/#26/#27):
      - changed content at the same chunk id was silently kept stale
      - a chunk-config change collided on UNIQUE(document_id, chunk_index)
        and dead-lettered the document
      - a shorter re-ingest left stale trailing chunks retrievable forever

    Semantics now, per incoming chunk id:
      unchanged (same content_sha256)  -> skip entirely
      changed                          -> update row + vectors, fix BM25 deltas
      new                              -> insert row + vectors + stats
    and any EXISTING id not in the incoming set is stale: deleted, BM25
    decremented, and tombstoned for Qdrant removal.

    Deadlock safety (findings #4/#29): all BM25 df deltas for the whole
    document are aggregated first and applied in ONE pass sorted by term_id,
    so concurrent workers always acquire term-row locks in the same order;
    the bm25_corpus singleton is touched exactly once, last.
    """
    pool = await get_pool()
    result = {"inserted": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    df_delta: dict[int, list] = {}          # term_id -> [term, delta]
    corpus_chunks_delta = 0
    corpus_len_delta = 0

    def bump(term_ids, terms, sign):
        for tid, term in zip(term_ids, terms):
            entry = df_delta.setdefault(int(tid), [term, 0])
            entry[1] += sign
            if term and not entry[0]:
                entry[0] = term

    async with pool.acquire() as conn, conn.transaction():
        # snapshot + lock this document's rows: serializes against concurrent
        # re-ingest/delete of the same document (FOR UPDATE OF c)
        existing = {r["id"]: r for r in await conn.fetch(
            """SELECT c.id, c.content_sha256, v.sparse_tf, v.doc_len
               FROM document_chunks c
               LEFT JOIN chunk_vectors v ON v.chunk_id = c.id
               WHERE c.document_id = $1 FOR UPDATE OF c""", document_id)}
        incoming_ids = {ch["id"] for ch in chunks}

        # ---- stale rows: delete + decrement + tombstone -------------------
        stale = [cid for cid in existing if cid not in incoming_ids]
        if stale:
            for cid in stale:
                row = existing[cid]
                bump([int(t) for t in (row["sparse_tf"] or {})],
                     [""] * len(row["sparse_tf"] or {}), -1)
                corpus_len_delta -= row["doc_len"] or 0
                corpus_chunks_delta -= 1
            await conn.execute(
                "DELETE FROM document_chunks WHERE id = ANY($1)", stale)
            await conn.executemany(
                "INSERT INTO vector_deletes (chunk_id) VALUES ($1)",
                [(c,) for c in stale])
            result["deleted"] = len(stale)

        # ---- incoming rows ------------------------------------------------
        for ch in chunks:
            old = existing.get(ch["id"])
            if old is not None and old["content_sha256"] == ch["content_sha256"]:
                result["unchanged"] += 1
                continue

            if old is not None:
                # changed content at the same id: retract the old text's stats
                bump([int(t) for t in (old["sparse_tf"] or {})],
                     [""] * len(old["sparse_tf"] or {}), -1)
                corpus_len_delta -= old["doc_len"] or 0
                await conn.execute(
                    """UPDATE document_chunks SET
                           chunk_index=$2, section_chunk_index=$3, text=$4,
                           context_header=$5, section=$6, section_title=$7,
                           is_narrative=$8, token_count=$9, char_count=$10,
                           content_sha256=$11, tenant=$12, region=$13,
                           embedding_model=$14, chunk_config=$15
                       WHERE id=$1""",
                    ch["id"], ch["chunk_index"], ch["section_chunk_index"],
                    ch["text"], ch["context_header"], ch["section"],
                    ch["section_title"], ch["is_narrative"], ch["token_count"],
                    ch["char_count"], ch["content_sha256"], ch["tenant"],
                    ch["region"], ch["embedding_model"], ch["chunk_config"])
                result["updated"] += 1
            else:
                await conn.execute(
                    """INSERT INTO document_chunks
                           (id, document_id, chunk_index, section_chunk_index, text,
                            context_header, section, section_title, is_narrative,
                            token_count, char_count, content_sha256, tenant, region,
                            embedding_model, chunk_config)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)""",
                    ch["id"], document_id, ch["chunk_index"],
                    ch["section_chunk_index"], ch["text"], ch["context_header"],
                    ch["section"], ch["section_title"], ch["is_narrative"],
                    ch["token_count"], ch["char_count"], ch["content_sha256"],
                    ch["tenant"], ch["region"], ch["embedding_model"],
                    ch["chunk_config"])
                result["inserted"] += 1
                corpus_chunks_delta += 1

            await conn.execute(
                """INSERT INTO chunk_vectors (chunk_id, dense, sparse_tf, doc_len, model)
                   VALUES ($1,$2,$3,$4,$5)
                   ON CONFLICT (chunk_id) DO UPDATE
                       SET dense = EXCLUDED.dense, sparse_tf = EXCLUDED.sparse_tf,
                           doc_len = EXCLUDED.doc_len, model = EXCLUDED.model""",
                ch["id"], ch["dense"],
                {str(k): v for k, v in ch["sparse_tf"].items()},
                ch["doc_len"], ch["embedding_model"])
            bump(list(ch["sparse_tf"].keys()), ch["sparse_terms"], +1)
            corpus_len_delta += ch["doc_len"]
            if ch["dense"] is not None:
                # a point with no dense vector cannot be upserted; under the
                # defer backend the backfill loader enqueues the outbox row
                # after the H100 vectors land
                await conn.execute(
                    "INSERT INTO vector_outbox (chunk_id) VALUES ($1) ON CONFLICT DO NOTHING",
                    ch["id"])

        # ---- BM25 deltas: one pass, sorted by term_id, then corpus once ---
        deltas = [(tid, term, n) for tid, (term, n) in sorted(df_delta.items()) if n != 0]
        if deltas:
            await conn.executemany(
                """INSERT INTO bm25_stats (term_id, term, df)
                   VALUES ($1, $2, greatest(0, $3))
                   ON CONFLICT (term_id) DO UPDATE
                       SET df = greatest(0, bm25_stats.df + $3)""",
                deltas)
        if corpus_chunks_delta or corpus_len_delta:
            await conn.execute(
                "UPDATE bm25_corpus SET n_chunks = greatest(0, n_chunks + $1),"
                " total_len = greatest(0, total_len + $2) WHERE id = 1",
                corpus_chunks_delta, corpus_len_delta)
    return result


# ---------------------------------------------------------------------------
# outbox
# ---------------------------------------------------------------------------

async def claim_pending_outbox(batch_size: int, max_attempts: int) -> list[dict]:
    """CLAIM a batch of pending rows (status -> 'syncing') and return them
    with everything the Qdrant point needs.

    The claim matters (review #3): an unclaimed batch could still be in
    flight while DELETE /documents purged the same chunks, and the late
    upsert would resurrect just-purged points. With claim + tombstone
    ordering (upserts drain before deletes each cycle) the system converges.
    attempts < max_attempts is the dead-letter gate (review #6): a poison
    row stops being retried instead of head-of-line blocking the sync forever.
    """
    pool = await get_pool()
    rows = await pool.fetch(
        """WITH claimed AS (
               SELECT id FROM vector_outbox
               WHERE status = 'pending' AND attempts < $2
               ORDER BY id LIMIT $1
               FOR UPDATE SKIP LOCKED)
           UPDATE vector_outbox o SET status = 'syncing'
           FROM claimed WHERE o.id = claimed.id
           RETURNING o.id""",
        batch_size, max_attempts,
    )
    if not rows:
        return []
    ids = [r["id"] for r in rows]
    details = await pool.fetch(
        """SELECT o.id AS outbox_id, o.attempts,
                  c.id AS chunk_id, c.document_id, c.chunk_index, c.text,
                  c.section, c.section_title, c.is_narrative, c.token_count,
                  c.tenant, c.region, c.embedding_model,
                  d.name AS document_name, d.gcs_uri, d.meta AS doc_meta,
                  d.tags AS doc_tags,
                  v.dense, v.sparse_tf, v.doc_len
           FROM vector_outbox o
           JOIN document_chunks c ON c.id = o.chunk_id
           JOIN documents d       ON d.id = c.document_id
           JOIN chunk_vectors v   ON v.chunk_id = c.id
           WHERE o.id = ANY($1)
           ORDER BY o.id""",
        ids,
    )
    return [dict(r) for r in details]


async def mark_outbox(outbox_ids: list[int], status: str, error: str | None = None,
                      max_attempts: int = 10) -> None:
    """'synced' finalizes; 'pending' re-queues with attempts+1, flipping to
    'failed' (dead letter) once attempts are exhausted."""
    pool = await get_pool()
    if status == "synced":
        await pool.execute(
            """UPDATE vector_outbox SET status = 'synced', synced_at = now()
               WHERE id = ANY($1)""", outbox_ids)
    else:
        await pool.execute(
            """UPDATE vector_outbox SET
                   attempts = attempts + 1, last_error = $2,
                   status = CASE WHEN attempts + 1 >= $3 THEN 'failed'
                                 ELSE 'pending' END
               WHERE id = ANY($1)""",
            outbox_ids, error, max_attempts)


async def reset_stale_syncing(older_than_minutes: int = 5) -> int:
    """Rows stuck in 'syncing' mean a worker died mid-batch; re-queue them."""
    pool = await get_pool()
    result = await pool.execute(
        """UPDATE vector_outbox SET status = 'pending'
           WHERE status = 'syncing'
             AND created_at < now() - make_interval(mins => $1)""",
        older_than_minutes)
    return int(result.split()[-1])


async def claim_pending_deletes(batch_size: int, max_attempts: int) -> list[dict]:
    """Tombstones waiting to be purged from Qdrant (written by
    delete_document and the reconcile in insert_chunks)."""
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT id, chunk_id FROM vector_deletes
           WHERE status = 'pending' AND attempts < $2
           ORDER BY id LIMIT $1
           FOR UPDATE SKIP LOCKED""",
        batch_size, max_attempts,
    )
    return [dict(r) for r in rows]


async def mark_deletes(ids: list[int], status: str, max_attempts: int = 10) -> None:
    pool = await get_pool()
    if status == "synced":
        await pool.execute(
            "UPDATE vector_deletes SET status='synced', synced_at=now() WHERE id = ANY($1)",
            ids)
    else:
        await pool.execute(
            """UPDATE vector_deletes SET attempts = attempts + 1,
                   status = CASE WHEN attempts + 1 >= $2 THEN 'failed' ELSE 'pending' END
               WHERE id = ANY($1)""",
            ids, max_attempts)


async def requeue_all_chunks() -> int:
    """Full re-sync: one pending outbox row for every chunk. This is the
    'drop the collection and rebuild it from Postgres' lever that makes every
    index experiment safe."""
    pool = await get_pool()
    result = await pool.execute(
        """INSERT INTO vector_outbox (chunk_id)
           SELECT id FROM document_chunks
           ON CONFLICT DO NOTHING"""
    )
    return int(result.split()[-1])


# ---------------------------------------------------------------------------
# BM25 statistics for query-time IDF
# ---------------------------------------------------------------------------

async def corpus_stats() -> dict:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT n_chunks, total_len FROM bm25_corpus WHERE id = 1")
    n = row["n_chunks"] or 0
    return {"n_chunks": n, "avgdl": (row["total_len"] / n) if n else 0.0}


async def term_dfs(term_ids: list[int]) -> dict[int, int]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT term_id, df FROM bm25_stats WHERE term_id = ANY($1)", term_ids)
    return {r["term_id"]: r["df"] for r in rows}


# ---------------------------------------------------------------------------
# citations + gauges + evals
# ---------------------------------------------------------------------------

async def chunks_by_ids(chunk_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT c.*, d.name AS document_name, d.meta AS doc_meta
           FROM document_chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.id = ANY($1)""",
        chunk_ids,
    )
    return {r["id"]: dict(r) for r in rows}


async def queue_depths() -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(
        """SELECT
             (SELECT count(*) FROM ingestion_jobs WHERE status = 'pending')   AS jobs_pending,
             (SELECT count(*) FROM vector_outbox  WHERE status = 'pending')   AS outbox_pending"""
    )
    return dict(row)


async def create_eval_run(config_name: str, index_params: dict) -> uuid.UUID:
    pool = await get_pool()
    row = await pool.fetchrow(
        "INSERT INTO evaluation_runs (config_name, index_params) VALUES ($1,$2) RETURNING id",
        config_name, index_params,
    )
    return row["id"]


async def finish_eval_run(run_id: uuid.UUID, status: str, metrics: dict) -> None:
    pool = await get_pool()
    await pool.execute(
        "UPDATE evaluation_runs SET status=$2, metrics=$3, finished_at=now() WHERE id=$1",
        run_id, status, metrics,
    )


async def add_eval_results(run_id: uuid.UUID, results: list[tuple[str, dict]]) -> None:
    pool = await get_pool()
    await pool.executemany(
        "INSERT INTO evaluation_results (run_id, query, metrics) VALUES ($1,$2,$3)",
        [(run_id, q, m) for q, m in results],
    )


async def get_eval_run(run_id: uuid.UUID) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM evaluation_runs WHERE id = $1", run_id)
    if not row:
        return None
    results = await pool.fetch(
        "SELECT query, metrics FROM evaluation_results WHERE run_id = $1", run_id)
    return {**dict(row), "results": [dict(r) for r in results]}
