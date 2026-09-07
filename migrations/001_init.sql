-- 001_init.sql — the entire rag_db schema (PRD 2.4).
--
-- Postgres is the SOURCE OF TRUTH. Qdrant is a derived index that can be
-- dropped and rebuilt from these tables at any time. That is why vectors are
-- persisted here (chunk_vectors) and not only in Qdrant.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid

-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name         text NOT NULL,
    source_type  text NOT NULL DEFAULT 'html',       -- html|pdf|md|txt|docx
    gcs_uri      text,
    tenant       text NOT NULL DEFAULT 'default',
    region       text NOT NULL DEFAULT 'us',
    tags         jsonb NOT NULL DEFAULT '[]',
    meta         jsonb NOT NULL DEFAULT '{}',        -- ticker, cik, fiscal_year, ...
    status       text NOT NULL DEFAULT 'pending',    -- pending|processing|ready|failed
    error        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_tenant_idx ON documents (tenant);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);

-- ---------------------------------------------------------------------------
-- chunk id is the DETERMINISTIC uuid5 minted by chunker.py, so re-ingesting
-- the same document under the same config upserts instead of duplicating.
CREATE TABLE IF NOT EXISTS document_chunks (
    id                   uuid PRIMARY KEY,
    document_id          uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index          int  NOT NULL,
    section_chunk_index  int  NOT NULL DEFAULT 0,
    text                 text NOT NULL,
    context_header       text NOT NULL DEFAULT '',
    page_number          int,
    section              text,                        -- 10-K item id, e.g. '1A'
    section_title        text,
    is_narrative         boolean NOT NULL DEFAULT false,
    token_count          int  NOT NULL,
    char_count           int  NOT NULL,
    content_sha256       text NOT NULL,
    tenant               text NOT NULL DEFAULT 'default',
    region               text NOT NULL DEFAULT 'us',
    embedding_model      text,
    chunk_config         jsonb NOT NULL DEFAULT '{}',
    created_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tenant_idx   ON document_chunks (tenant);

-- ---------------------------------------------------------------------------
-- Vectors live in Postgres so Qdrant can always be rebuilt (and so the Modal
-- H100 backfill has somewhere durable to write results). dense is float4[];
-- sparse_tf holds RAW term frequencies {term_id: tf} — the BM25 weighting is
-- applied at Qdrant-sync time so it always uses current corpus statistics.
CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id      uuid PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
    dense         real[],
    sparse_tf     jsonb,                              -- {"123456": 3, ...}
    doc_len       int,                                -- BM25 tokens in this chunk
    model         text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    status       text NOT NULL DEFAULT 'pending',  -- pending|processing|done|failed|dead
    attempts     int  NOT NULL DEFAULT 0,
    last_error   text,
    worker_id    text,
    started_at   timestamptz,
    finished_at  timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON ingestion_jobs (status);

-- ---------------------------------------------------------------------------
-- Transactional outbox: chunk rows and outbox rows commit in ONE transaction,
-- then a separate worker drains pending rows into Qdrant. If Qdrant is down
-- nothing is lost — the rows wait. rag_outbox_pending_total watches this.
CREATE TABLE IF NOT EXISTS vector_outbox (
    id          bigserial PRIMARY KEY,
    chunk_id    uuid NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    status      text NOT NULL DEFAULT 'pending',   -- pending|synced|failed
    attempts    int  NOT NULL DEFAULT 0,
    last_error  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    synced_at   timestamptz
);
CREATE INDEX IF NOT EXISTS outbox_status_idx ON vector_outbox (status);
CREATE UNIQUE INDEX IF NOT EXISTS outbox_chunk_pending_uq
    ON vector_outbox (chunk_id) WHERE status = 'pending';

-- ---------------------------------------------------------------------------
-- BM25 corpus statistics (PRD 3.4: IDF comes from rag_db, not from Qdrant).
-- df = number of chunks containing the term. Updated inside the ingestion
-- transaction so stats can never drift from the chunks they describe.
CREATE TABLE IF NOT EXISTS bm25_stats (
    term_id  bigint PRIMARY KEY,
    term     text NOT NULL,
    df       int  NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bm25_corpus (
    id          int PRIMARY KEY DEFAULT 1 CHECK (id = 1),   -- singleton row
    n_chunks    bigint NOT NULL DEFAULT 0,
    total_len   bigint NOT NULL DEFAULT 0
);
INSERT INTO bm25_corpus (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    config_name   text NOT NULL,
    index_params  jsonb NOT NULL DEFAULT '{}',
    status        text NOT NULL DEFAULT 'running',   -- running|done|failed
    metrics       jsonb NOT NULL DEFAULT '{}',       -- recall@k, mrr, ndcg, latency
    created_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id       bigserial PRIMARY KEY,
    run_id   uuid NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    query    text NOT NULL,
    metrics  jsonb NOT NULL DEFAULT '{}'
);

COMMIT;
