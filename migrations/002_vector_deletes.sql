-- 002_vector_deletes.sql
--
-- Findings #2/#3/#20/#35 (adversarial review): Qdrant deletes were fired
-- after the Postgres commit with nothing durable recording them — a crash or
-- Qdrant outage in that window orphaned points forever (and a retried DELETE
-- hit the 404 branch, so they could never be purged).
--
-- Fix: delete TOMBSTONES, written in the same transaction that removes the
-- chunks. The outbox worker drains them into Qdrant deletes after upserts.
-- Deliberately NO foreign key: these rows must outlive the chunk they purge.

BEGIN;

CREATE TABLE IF NOT EXISTS vector_deletes (
    id          bigserial PRIMARY KEY,
    chunk_id    uuid NOT NULL,
    status      text NOT NULL DEFAULT 'pending',   -- pending|synced|failed
    attempts    int  NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now(),
    synced_at   timestamptz
);
CREATE INDEX IF NOT EXISTS vector_deletes_status_idx ON vector_deletes (status);

COMMIT;
