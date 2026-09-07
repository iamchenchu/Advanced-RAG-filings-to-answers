# Advanced-RAG-System — End-to-End Explainer

*What every file does, how data flows through them, and exactly what to run.*
*Written 2026-08-18; updated 2026-08-19 after the FULL PRODUCTION BACKFILL:
6,947 filings in GCS, 1,736,942 chunks, all embedded with BGE-M3 on Modal H100s,
fully searchable with measured recall/latency curves and year-aware golden evals.*

---

## 1. What this system is, in one picture

```
                          INGESTION (write path)
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  SEC EDGAR ──► edgar_downloader.py ──► GCS (raw .htm)                │
  │                                          │                           │
  │  POST /api/v1/documents ──► documents + ingestion_jobs rows (202)    │
  │                                          │                           │
  │  ingestion_worker.py  ── claims job (FOR UPDATE SKIP LOCKED)         │
  │      parser.py        ── HTML/iXBRL ► clean text ► Item sections     │
  │      chunker.py       ── sections ► 512-token windows, 64 overlap    │
  │      embedder client  ── dense vectors (fastembed local / TEI prod)  │
  │      sparse.py        ── BM25 term counts                            │
  │      ONE TRANSACTION  ── chunks + vectors + bm25 stats + outbox      │
  │                                          │                           │
  │  outbox_worker.py     ── pending outbox ► Qdrant points              │
  │                          (into BOTH `chunks` and `chunks_flat`)      │
  └──────────────────────────────────────────────────────────────────────┘

                          QUERY (read path)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  POST /api/v1/query                                                  │
  │    rewrite (optional LLM) ► embed query                              │
  │      ├── DENSE  search   Qdrant HNSW      top 100                    │
  │      └── SPARSE search   Qdrant BM25      top 100                    │
  │    RRF fusion (hand-written) ► MMR diversity ► hydrate text from PG  │
  │    ► cross-encoder RERANK ► context builder (budget, dedupe,         │
  │      citation ids) ► generate (LLM, optional) ► answer + citations   │
  └──────────────────────────────────────────────────────────────────────┘
```

Two facts organize everything:

1. **Postgres is the source of truth; Qdrant is a derived index.** Chunk text,
   vectors, and BM25 statistics all live in `rag_db`. Qdrant can be dropped and
   rebuilt from Postgres at any time (`outbox_worker.py --resync`). This is what
   makes every index experiment safe.
2. **Ingestion is a resumable data pipeline, not an API call.** Every step is
   idempotent: deterministic chunk ids, `ON CONFLICT DO NOTHING` inserts,
   upsert-by-id in Qdrant. Kill any worker at any line and re-run it — nothing
   is lost, nothing is doubled. (This was tested live: re-ingesting the same
   filing inserted 0 new chunks.)

---

## 2. File-by-file, in the order data flows

### Ingestion

| File | What it does | What to check when reviewing |
|---|---|---|
| `app/ingestion/edgar_downloader.py` | SEC submissions API → find latest 10-K → download → upload to GCS. Returns metadata (cik, accession, date). | The three steps are separate functions so each is swappable. `SEC_USER_AGENT` is required by SEC's fair-use policy. |
| `app/ingestion/parser.py` | GCS/local `.htm` → BeautifulSoup flatten (iXBRL included) → regex-split into the 21 10-K Items. Takes the **last** match of each Item header to skip the table of contents. | `ITEMS` regexes are tuned to Apple's wording; widening them is a known TODO before multi-company scale. |
| `app/ingestion/chunker.py` | Sections → sentence-aware token windows (512 budget, 64 overlap), measured with the **real embedding model's tokenizer** — never estimated. Strips page furniture. Mints **deterministic chunk ids**: `uuid5(source_uri + item + index + config-fingerprint)`. | The config fingerprint inside the id: change `CHUNK_SIZE_TOKENS` and you get a *new id space* instead of silently overwriting old vectors. `text` (clean, quoted in citations) vs `embed_text` (context header prepended, only for embedding). |
| `app/clients/embedder.py` | One interface, three backends: `tei` (HTTP, prod), `fastembed` (in-process ONNX, local dev — real semantics, no torch), `hash` (deterministic garbage, tests). L2-normalizes everything. | The invariant: **queries and documents must share one model.** `embedding_model` is recorded per chunk row so stale vectors are findable after a model switch. |
| `app/retrieval/sparse.py` | Hand-written BM25 math, pure functions: analyzer, stable 31-bit term ids (blake2b, not salted `hash()`), tf encoding, index-time weights, query-time IDF. | The split: tf-half applied at Qdrant-sync time, IDF-half at query time from live `rag_db` stats — so rarity is always current. Unit-tested against textbook Okapi BM25. |

### Persistence

| File | What it does | What to check |
|---|---|---|
| `migrations/001_init.sql` | The whole `rag_db` schema: `documents`, `document_chunks`, `chunk_vectors`, `ingestion_jobs`, `vector_outbox`, `bm25_stats` + `bm25_corpus`, `evaluation_runs/results`. | `chunk_vectors` is why Qdrant is rebuildable. The partial unique index on `vector_outbox (chunk_id) WHERE status='pending'` — max one pending sync per chunk. |
| `app/persistence/db.py` | asyncpg pool + a 20-line migration runner (tracked in `schema_migrations`). | jsonb codec is registered once here so repositories never `json.dumps` by hand. |
| `app/persistence/repository.py` | **Every SQL statement in the system.** The two patterns worth reading: `claim_job` (FOR UPDATE SKIP LOCKED — a work queue in 12 lines, N workers never collide) and `insert_chunks` (chunks + vectors + BM25 stats + outbox in ONE transaction). | `delete_document` decrements BM25 `df` for every removed chunk *inside the same transaction* — corpus stats can never describe chunks that don't exist. `create_document` dedupes on `(gcs_uri, tenant)`. |

### Workers (each is a small CLI; `--once` drains and exits, no flag = poll forever)

| File | What it does | Crash story |
|---|---|---|
| `workers/ingestion_worker.py` | claim → load → parse → chunk → embed → sparse-encode → one-transaction insert → mark done. CPU work runs in a thread so the event loop stays live. | Dies mid-job → job stays `processing` → reaper resets it → retry re-runs everything → deterministic ids make the re-insert a no-op for whatever already landed. |
| `workers/outbox_worker.py` | Drains `vector_outbox` → computes BM25 index weights with *current* avgdl → upserts points into **both** collections. `--resync` re-queues every chunk (the rebuild lever). | Qdrant down → rows stay pending, gauge grows, nothing lost. Point id = chunk id → double-sync = same point. |
| `workers/reaper.py` | Resets jobs stuck in `processing` past the timeout; dead-letters at max attempts (a poisoned document must not eat the worker pool forever). | |

### Retrieval (query path — `app/retrieval/`)

| File | Stage | The idea |
|---|---|---|
| `dense.py` | semantic search | `ef_search` rides on every query (no rebuild to tune it — that's Experiment 1). `exact=True` + the flat collection = ground truth mode. |
| `sparse_search.py` | keyword search | Query IDF from `rag_db` stats → Qdrant sparse dot product = BM25. Returns `[]` honestly when no term matches. |
| `fusion.py` | RRF (hand-written) | `Σ 1/(60+rank)`. Ranks, not scores — cosine ~0.8 and BM25 ~21 can't be added; ranks are scale-free. `weighted` mode exists for A/B. |
| `mmr.py` | diversity | Greedy `λ·relevance − (1−λ)·max-sim-to-selected` over the dense vectors that came back **with** the search (zero extra embed calls). Kills the near-duplicates that 64-token overlap guarantees. |
| `pipeline.py` | orchestrator | Every stage timed (Prometheus + `timings_ms`), every failure **degrades instead of erroring**: embedder down → sparse-only; reranker down → fusion order; LLM down → `answer: null` + citations intact. Also the **hydrate** step: Qdrant payloads are lean; the ~20 MMR survivors get their text from Postgres in one keyed lookup — which doubles as citation-integrity (a point with no chunk row is dropped, never cited). |
| `../clients/reranker.py` | rerank | Cross-encoder reads query+passage *together* — actual relevance, not topic proximity. Too slow for the corpus, cheap for the top-100. Scores recorded for debugging. |
| `context_builder.py` | context + citations | In order: near-dup suppression (cosine ≥ 0.95), per-document cap (3), hard token budget (prevented, not truncated), citation ids `[1..n]` assigned here **and only here** — so every id resolves by construction. |
| `../clients/inference.py` | generate | OpenAI-compatible HTTP to Distributed-Inference-Serving, `X-Workload-Class: rag` header, circuit breaker, retries only before first output. Every failure → `InferenceUnavailable` → degradation contract. |

### API (`app/api/` + `app/main.py`)

| File | Endpoints | Notes |
|---|---|---|
| `query.py` | `POST /api/v1/query` | **The contract Multi-Agent consumes.** `generate: false` returns passages with no LLM call — the single most important contract decision. |
| `documents.py` | `POST/GET /documents`, `GET .../status`, `DELETE` | POST answers 202 immediately; ingestion is minutes of work and never blocks a request. DELETE cascades in PG *and* purges Qdrant. |
| `evaluations.py` | `POST/GET /evaluations` | Runs the eval harness as a background task, results in `rag_db`. |
| `health.py` | `/health` `/ready` `/metrics` | `/ready` fails fast when Postgres is down (pod leaves the Service); everything else degrades instead. |
| `schemas.py` | pydantic models | These models *are* `openapi.json`. `extra="forbid"` on filters: unknown keys → 400 `invalid_filter`, never a silent no-op. |
| `main.py` | assembly | Startup: pool → migrations → Qdrant collections (all idempotent) → gauge-refresher task (the two silent-failure gauges). Serves the debug UI at `/`. |

### Evaluation (`evaluation/`)

| File | What it measures |
|---|---|
| `run_eval.py` | Recall@K / MRR of HNSW **against the flat exact collection** + latency percentiles. Results → `evaluation/results/*.json` + `rag_db`. Recall@K is undefined without the flat ground truth — that's why `chunks_flat` exists. |
| `experiments/ef_sweep.py` | **Experiment 1 (PRD: do this first).** Sweeps `ef_search` 16→512, no rebuilds, produces the recall-vs-latency curve. |
| `build_ground_truth.py` | Verifies the two collections agree; re-queues everything from Postgres if not. |
| `fixtures/queries.jsonl` | 20 finance queries. Recall-vs-flat needs no relevance labels — it isolates ANN approximation error, not embedding quality. |

### Scale-out (`modal_app/` + `scripts/`) — the H100 path you asked for

The 5,000-filing backfill (~1.5M chunks) does **not** embed locally. Three
resumable stages, GCS as the handoff:

```
scripts/make_shards.py      Postgres -> GCS      chunks missing vectors for the
                                                 target model, ~2k-chunk shards
modal_app/embed_backfill.py GCS -> H100 -> GCS   modal run ...; one container per
                                                 shard (8 parallel), BGE-M3 fp16,
                                                 token-sorted batches, .done markers
scripts/load_embeddings.py  GCS -> Postgres      upsert chunk_vectors + outbox rows
workers/outbox_worker.py    Postgres -> Qdrant   same worker as always
```

GPU dies at shard 350/750 → re-run `modal run`; done-markers skip finished
shards, the loader upserts idempotently, the outbox dedupes. **Nothing restarts
from zero.** The `embedding_model` column is how `make_shards.py` finds work:
"chunks whose vector is missing *or built by a different model*."

### Frontend

`frontend/index.html` — one static file served at `/`. Query box, answer,
citations, per-stage timing bars, and the full debug panel (dense/sparse/fusion/
MMR/rerank tables). This satisfies PRD 3.9; the full Next.js app is future work.

---

## 3. Local dev vs production

| | Local (this machine, verified today) | Production |
|---|---|---|
| Postgres | homebrew `postgresql@18`, db `rag_db` | `postgres:16` (compose/k8s) |
| Qdrant | docker container `rag-qdrant` | `qdrant/qdrant` |
| Dense embeddings | `fastembed` in-process, `bge-small-en-v1.5` (384-d) | TEI serving `bge-m3` (1024-d); bulk via **Modal H100** |
| Reranker | `fastembed` cross-encoder `ms-marco-MiniLM-L-6-v2` | TEI `bge-reranker-v2-m3` |
| Generation | **degraded by design** (no LLM running) — or `ollama serve` + point `INFERENCE_URL` at `http://localhost:11434/v1` | Distributed-Inference-Serving |
| Sparse BM25 | identical everywhere — pure Python + rag_db stats | identical |

Switching models later is designed-for, not feared: chunk ids fingerprint the
config, `embedding_model` marks every vector row, `make_shards.py` finds stale
rows, and the flat collection rebuilds from Postgres.

`docker-compose.yml` contains the full-fidelity stack (TEI containers included)
when you want to exercise the HTTP client paths.

---

## 3.4 The production backfill — what actually ran (2026-08-19)

The full pipeline executed overnight, unattended:

| Stage | Result |
|---|---|
| Download (EDGAR → GCS) | **6,947 filings** from **872 companies** (top ~1,150 tickers attempted; non-10-K filers drop out), FY2008–2026, 0 unresolved failures, resumable throughout |
| Parse + chunk (6 workers, defer backend) | **1,736,942 chunks** with BM25 term counts, dense=NULL awaiting GPU |
| Shard export → Modal volume | 867 shards, 3.5 GB |
| **H100 fleet** (8 containers, BGE-M3 fp16, token-sorted batches) | all shards embedded, 0 errors, ~$30 |
| Load → Postgres → outbox → Qdrant | 1.73M points in BOTH collections, 0 failed, index green |
| Query-time embedding | BAAI's official ONNX export, in-process, ~92 ms warm — verified **cosine 1.0000** vs H100 vectors on identical text |

Lessons that are now encoded in the code: `.done` resume markers must be cleared when
shards are re-exported (a stale marker silently skipped 2 shards — caught by the
`missing vectors` count, fixed by re-running two shards); a detached orchestrator's
psql can flake where asyncpg doesn't (drive long pipelines through the DB-native path).

## 3.5 The corpus and the golden dataset

**Corpus (live now):** **872 companies** × up to 10 years of 10-Ks (FY2008–2026) = **6,947 documents,
1,736,942 chunks**, consistent three ways (Postgres = both Qdrant collections = BM25
counters), every chunk carrying a 1024-dim BGE-M3 vector.

**Parser generalization (measured, then fixed):** the first multi-company pass exposed
the Apple-tuned parser — MSFT had 0 Risk-Factor chunks (EDGAR wraps single words in
nested spans; `separator=" "` split "RISK" into "RIS K"), NVDA had 1 (a late
cross-reference beat the real header under last-match). Fixed with block-aware text
extraction plus an in-order chain-selection DP over all header matches (a 10-K always
presents Items in order; maximize items placed, tie-break toward the later chain to beat
the TOC). After: MSFT 34 risk chunks, NVDA 51, WMT MD&A 2→30, zero fallbacks anywhere.

**Golden dataset — the labeled one** (`evaluation/fixtures/golden.jsonl`): 18 natural
questions, each labeled with the chunk(s) containing its answer. Labels are
**content-anchored** (ticker + Item + must-contain phrases, resolved to ids by
`evaluation/build_golden.py`), so re-chunking never invalidates them — re-run the builder
and they re-resolve. This is different from the flat-collection eval: `run_eval.py`
isolates ANN approximation error; `run_golden.py` measures whether retrieval finds the
RIGHT passage, per config:

On the FULL 1.73M-chunk corpus with BGE-M3 (labels are year-aware — any fiscal year's
copy of the answer passage counts, since the corpus holds ~10 near-identical filings per
company):

| config | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| dense (BGE-M3) | 0.444 | 0.611 | **0.778** | 0.561 |
| sparse (hand-written BM25) | 0.056 | 0.167 | 0.333 | 0.123 |
| hybrid (RRF) | 0.333 | **0.667** | **0.778** | 0.458 |
| full (+MMR + MiniLM rerank) | 0.444 | 0.444 | 0.667 | 0.487 |

Context: the earlier 0.722 hit@10 was on 2,275 chunks; **0.778 is on a 760× larger
corpus** — quality went UP as the problem got harder. Two quantified findings the
per-stage design exists to surface: (1) BM25 degrades at this scale — ten years of
near-identical boilerplate per company crowd the term space (it still owns exact-term
queries: tickers, case names, dollar figures); (2) the small English MiniLM reranker now
HURTS vs raw dense at hit@5 — the measured case for `bge-reranker-v2-m3` on TEI in
production. 18 queries remains smoke-test-sized; grow `build_golden.py`'s SPEC.

**Experiment 1 (ef_search sweep), finally on real scale:**

| ef_search | recall@10 vs flat | p50 | p95 |
|---|---|---|---|
| 16 | 0.910 | 463 ms | 633 ms |
| **32 (new default)** | **0.985** | 726 ms | 840 ms |
| 64 | 0.990 | 1.2 s | 1.5 s |
| 128 | 1.000 | 2.2 s | 2.6 s |

A point of recall costs ~0.5–1 s on cold on-disk vectors. To reach the PRD's
p95 < 400 ms: scalar INT8 quantization with `always_ram` is now enabled on the HNSW
collection (requantizing in the background; the flat ground truth stays full-precision)
— re-run the sweep after it settles to measure the gain (Experiment 3's territory).

## 4. What to run

### One-time setup

```bash
cd Advanced-RAG-System
python -m pip install -r requirements.txt        # (already installed in ../.venv)

# Postgres role + database (already done on this machine):
psql -U postgres -c "CREATE ROLE rag LOGIN PASSWORD 'changeme'"
psql -U postgres -c "CREATE DATABASE rag_db OWNER rag"

# Qdrant (already running as container rag-qdrant):
docker run -d --name rag-qdrant -p 6333:6333 -p 6334:6334 \
  -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant

python scripts/init_db.py                        # migrations + both collections
```

### The daily loop (order matters)

```bash
# 1. API (terminal 1) — serves the UI at http://localhost:8000
uvicorn app.main:app --port 8000 --loop asyncio

# 2. Register a document (terminal 2) -> 202 + job row
python scripts/seed.py

# 3. Ingest: parse -> chunk -> embed -> Postgres (+outbox)
python workers/ingestion_worker.py --once

# 4. Sync: outbox -> Qdrant (both collections)
python workers/outbox_worker.py --once

# 5. Query — retrieval only (the generate:false contract):
curl -s http://localhost:8000/api/v1/query -H 'Content-Type: application/json' \
  -d '{"query":"What are Apples main supply chain risks?","top_k":3,"generate":false}' | jq

# ...or open http://localhost:8000/ and use the debug console.
```

### Quality gates

```bash
python -m pytest tests/unit -q                    # 26 tests: BM25/RRF/MMR/context/chunker/parser
python evaluation/run_eval.py --config baseline   # Recall@K vs flat + latency (ANN error)
python evaluation/build_golden.py                 # re-resolve golden labels vs the corpus
python evaluation/run_golden.py                   # LABELED eval: dense vs sparse vs hybrid vs full
python evaluation/experiments/ef_sweep.py         # Experiment 1: the recall/latency curve
python evaluation/build_ground_truth.py           # set-diff repair: strays deleted, missing queued
python scripts/export_openapi.py                  # publish the rag-v1 contract
python workers/reaper.py --once                   # sweep stuck jobs (cron this)
```

### Growing the corpus

```bash
python scripts/download_corpus.py --tickers AVGO ORCL COST --years 3   # any tickers
python workers/ingestion_worker.py --once                              # (run 2+ in parallel — SKIP LOCKED)
python workers/outbox_worker.py --once
```

### The H100 backfill (when the 5,000-filing corpus lands)

```bash
pip install modal && modal setup
modal secret create gcp-credentials GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat service-account.json)"

python scripts/make_shards.py  --model BAAI/bge-m3 --prefix embed_jobs/$(date +%F)
modal run modal_app/embed_backfill.py --shard-prefix gs://llm-platform-services-bucket/embed_jobs/$(date +%F)
python scripts/load_embeddings.py --prefix embed_jobs/$(date +%F)
python workers/outbox_worker.py --once
```

(Before that backfill: freeze chunking on ~500 filings, switch `.env` to
`EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_DIM=1024`, `EMBEDDING_BACKEND=tei` —
and re-create the Qdrant collections since the dimension changes.)

---

## 5. What was verified live today (all on this machine)

| Check | Result |
|---|---|
| 22 unit tests (BM25 vs textbook, RRF rank-invariance, MMR diversity, context budget/cap/dedupe, chunker packing) | ✅ pass |
| Apple FY2025 10-K: GCS → parse → chunk → embed → 111 chunks in PG | ✅ |
| Outbox → Qdrant: 111 points in `chunks` **and** `chunks_flat` | ✅ |
| Hybrid query end-to-end, 455 ms total (rerank 348 of it), correct Item 1A supply-chain chunk at #1 | ✅ |
| Exact-term query "Epic Games lawsuit injunction" → Item 3, sparse rank 1 | ✅ |
| Metadata pre-filter `{"item":"7"}` returns only MD&A chunks | ✅ |
| Degradation: `generate:true` with inference down → `answer: null`, `degraded:["generation_unavailable"]`, citations intact, HTTP 200 | ✅ |
| Unknown filter key → HTTP 400 `invalid_filter` | ✅ |
| Idempotency: full re-ingest of same filing → **0 new chunks** | ✅ |
| Registration dedupe: re-seeding returns the same `document_id` | ✅ |
| DELETE cascades PG + purges Qdrant | ✅ |
| Recall@1/5/10 vs flat = 1.0, p50 2.2 ms (expected: 111 pts < 10k full-scan threshold → brute force; the *harness* is what's proven) | ✅ |
| ef_search sweep 16/64/256 runs, results committed | ✅ |
| `openapi.json` exported (20 KB) | ✅ |

Bugs found **by running** (all fixed): reranker crashed on missing chunk text →
added the hydrate stage; `answer: null` was being stripped from responses →
broke the degradation contract; unknown filter keys silently dropped; `json_agg`
returned strings through asyncpg; worker shutdown crashed on a second event
loop; duplicate document rows on re-registration.

**Adversarial review (59-agent fleet, every finding independently verified by a
refuter agent): 44 confirmed, 10 refuted — all 44 fixed.** The headline ones:

- *(critical)* re-ingesting changed content silently kept stale chunks, and a
  chunk-config change dead-lettered the document via a UNIQUE collision →
  `insert_chunks` is now a full **reconcile** (unchanged/changed/new/stale per
  chunk id, BM25 deltas applied sorted-by-term in one pass, corpus counter
  touched once). Proven live: the full-corpus re-ingest replaced all 2,254 old
  chunks in-transaction — the exact scenario that used to dead-letter.
- *(critical)* an over-selective filter returned 503 `retrieval_unavailable` —
  outage and honest-empty are now distinct (503 only when no retrieval PATH
  functioned).
- deletes now write **tombstones** in the same transaction (crash-safe Qdrant
  purge, drained by the outbox worker after upserts, so an in-flight sync can
  never permanently resurrect deleted points); outbox rows are claimed
  (`syncing`) with dead-lettering and poison-row isolation; BM25 updates are
  deadlock-ordered; jobs are worker-fenced against reaper races; fastembed
  backends now degrade instead of 500ing; blocking calls moved off the event
  loop; `tags` filter actually filters; `weighted` fusion no longer zeroes
  equal-score lists; `top_k` > 5 no longer silently capped; eval runs record
  failures instead of hanging in `running`.

---

## 6. Honest gaps (so your review knows where to push)

- **Parser now generalizes across the 10 test companies** (zero fallbacks), but
  JPM-style bank filings keep MD&A in an exhibit (1 chunk is genuine, not a bug),
  and 5,000 filings will surface more layouts — `sec-parser` still worth
  evaluating at that scale.
- **Item 8 tables** are flattened number-runs; they chunk correctly but embed
  poorly. Decide: special table handling or exclude from narrative retrieval
  (`is_narrative` filter already exists).
- **`stream: true` is accepted but not implemented** (needs a live LLM to build
  against; the field is in the contract so Multi-Agent's client is ready).
- **Auth is a stub** (`AUTH_ENABLED=false`); rate limiting off locally.
- **Query rewrite untested** (needs the LLM; switchable, currently off).
- **Recall-vs-flat numbers stay trivial below 10k points** (full-scan threshold);
  the labeled golden eval is the meaningful metric until the corpus crosses it.
- Experiments 2–5 (HNSW params, quantization, scaling, filter selectivity) are
  scaffolded by the same harness but not yet run.
