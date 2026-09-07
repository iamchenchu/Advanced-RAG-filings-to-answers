# Read_Code.md - how to read this codebase, flow by flow

A guided reading order for the source. Not WHAT the system is (that is
[explainer.md](explainer.md)) and not WHY each design was chosen (that is
[COMPONENTS.md](COMPONENTS.md)) - this file tells you WHICH file to open NEXT,
what to look for inside it, and how the calls chain together, so you can follow
a filing through the code the same way the bytes travel.

Line numbers are from the current commit; they drift as code changes - trust the
function NAMES first.

---

## The big picture: six flows, one support layer

```
FLOW 1  REGISTER   a filing becomes a row + a job          (seconds)
FLOW 2  INGEST     a job becomes chunks in Postgres        (the write path)
FLOW 3  SYNC       chunks become searchable Qdrant points  (the outbox)
FLOW 4  QUERY      a question becomes cited passages       (the read path)
FLOW 5  BACKFILL   1.7M chunks get GPU vectors             (the batch path)
FLOW 6  EVAL       retrieval quality becomes a number      (the proof)

SUPPORT  config, database pool, metrics, schemas - read pieces as needed
```

Suggested reading sessions: one flow per sitting, in the order above. Each flow
below ends with a "trace it yourself" exercise - do them; ten minutes of
watching real values beats an hour of reading.

---

## FLOW 1 - REGISTER: filing -> document row + job row

```
scripts/seed.py  (or scripts/download_corpus.py)
  |  HTTP POST /api/v1/documents          <- everything enters via the API
  v
app/api/documents.py :: upload_document()          L22
  |  validates DocumentIn (app/api/schemas.py)
  v
app/persistence/repository.py :: create_document() L30
  |  ONE transaction:
  |    - reuse-or-insert documents row  (dedupe on gcs_uri+tenant)
  |    - INSERT ingestion_jobs row      (status=pending)
  v
returns 202 {document_id, job_id}   <- API never does the heavy work
```

What to look for:
- `create_document` L30: the UPDATE-then-INSERT dedupe - re-registering the
  same gcs_uri refreshes metadata and enqueues a new job instead of duplicating.
- The 202 pattern: the request returns in milliseconds; minutes of parsing
  happen later, in FLOW 2, in a different process.

Trace it yourself:
```bash
python scripts/seed.py            # then:
psql "postgresql://rag:changeme@localhost:5432/rag_db" \
  -c "SELECT id,status FROM ingestion_jobs ORDER BY id DESC LIMIT 1"
```

---

## FLOW 2 - INGEST: job -> chunks in Postgres (the heart)

```
workers/ingestion_worker.py :: main() L177 -> _run() L159 -> work_loop() L107
  |  loop forever (or --once):
  v
repository.claim_job() L160
  |  "FOR UPDATE SKIP LOCKED"  <- THE queue trick: N workers, no collisions
  v
workers/ingestion_worker.py :: process_document() L48    (runs in a thread)
  |
  |--> app/ingestion/parser.py
  |      load_html()        L38   gs:// or local path -> HTML string
  |      html_to_text()     L59   block-aware flatten (the "RIS K" fix lives here)
  |      split_into_items() L113  the in-order chain DP over Item headers
  |
  |--> app/ingestion/chunker.py
  |      chunk_sections()   L391  the orchestrator, per Item:
  |        clean_section()  L156    strip page furniture
  |        split_sentences()L184    sentence boundaries
  |        pack()           L238    greedy 512-token windows + 64 overlap
  |        _hard_split()    L203    tables with no sentences -> token cuts
  |        Tokenizer        L103    REAL BGE-M3 token counts (never estimates)
  |      -> Chunk dataclass L317  (note: text vs embed_text, uuid5 id recipe)
  |
  |--> app/clients/embedder.py :: Embedder.embed()
  |      backend = onnx | fastembed | tei | defer | hash
  |      "defer" returns None -> chunks land with dense=NULL for FLOW 5
  |
  |--> app/retrieval/sparse.py :: encode_document() L56
  |      analyze() L45 tokenize -> term_id() L49 stable ids -> tf counts
  v
repository.insert_chunks() L232        <- read this one slowly, twice
  |  ONE transaction, full RECONCILE:
  |    unchanged (same content_sha256) -> skip
  |    changed                         -> update + fix BM25 stats
  |    new                            -> insert + outbox row
  |    stale (id no longer produced)  -> delete + tombstone
  |    BM25 deltas applied sorted-by-term (deadlock safety)
  v
repository.finish_job() L181   (worker_id fence vs the reaper)
```

What to look for:
- `work_loop` L107: what happens on exception - attempts, dead-letter, document
  status. This is the crash-safety story.
- `insert_chunks` L232: the four-way reconcile. This function is why you can
  kill any worker at any line and re-run safely.
- `workers/reaper.py`: 30 lines - resets jobs from dead workers.

Trace it yourself: re-run ingestion on the already-ingested corpus and watch
the reconcile report "unchanged" for everything:
```bash
python scripts/seed.py && python workers/ingestion_worker.py --once
# expect: "0 inserted, 0 updated, N unchanged, 0 stale removed"
```

---

## FLOW 3 - SYNC: Postgres outbox -> Qdrant points

```
workers/outbox_worker.py :: drain() L37
  |  reset_stale_syncing() L440   <- rows stuck 'syncing' = a worker died
  v
repository.claim_pending_outbox() L376
  |  claims a batch (status pending -> syncing), attempts < max = dead-letter gate
  v
app/retrieval/sparse.py :: index_weights() L71
  |  the TF half of BM25, computed with CURRENT avgdl
  v
app/clients/qdrant.py :: upsert_points() L106
  |  point id = chunk id (idempotent by construction)
  |  payload  = ticker/fiscal_year/section/... (NOT the text - see FLOW 4 hydrate)
  |  both collections: chunks (HNSW) and chunks_flat (ground truth)
  v
repository.mark_outbox() L421   synced | pending+attempts | failed
  then: claim_pending_deletes() L451 -> qdrant.delete_points() L150
        (tombstones AFTER upserts - ordering prevents resurrection)
```

What to look for:
- The batch-fails -> retry-row-by-row fallback in `drain()`: poison-row isolation.
- `ensure_collections()` L52 in qdrant.py: dimension guard + on_disk + quantization.

Trace it yourself:
```bash
psql ... -c "SELECT status,count(*) FROM vector_outbox GROUP BY 1"
python workers/outbox_worker.py --once
```

---

## FLOW 4 - QUERY: question -> cited passages (the read path)

```
app/api/query.py :: query() L29
  |  QueryRequest schema (schemas.py: extra="forbid" -> unknown filter = 400)
  v
app/retrieval/pipeline.py :: run_query() L63     <- the conductor, read top to bottom
  |  every stage wrapped in `with t.stage(...)` - ONE wrapper, THREE outputs:
  |  timings_ms (response) + Prometheus histogram + OpenTelemetry span
  |  (app/observability/tracing.py; view waterfalls at localhost:16686)
  |
  |  1 rewrite   clients/inference.py (optional; LLM = ollama llama3.2:3b locally)
  |  2 embed     clients/embedder.py :: embed_one()   ONNX BGE-M3, ~92ms
  |  3 dense     retrieval/dense.py :: dense_search()     - Qdrant HNSW top 100
  |    sparse    retrieval/sparse_search.py :: sparse_search()
  |                sparse.query_weights() L80  <- the IDF half, live from bm25_stats
  |    (both use qdrant.build_filter() L160    <- PRE-filtering)
  |    outage vs honest-empty logic: 503 only if NO path functioned
  |  4 fusion    retrieval/fusion.py :: rrf()       ranks, 1/(60+rank)
  |  5 mmr       retrieval/mmr.py :: mmr_select()   top 20, lambda=0.7
  |  5b hydrate  repository.chunks_by_ids()         text lives ONLY in Postgres
  |  6 rerank    clients/reranker.py :: rerank()    cross-encoder on ~20
  |  7 context   retrieval/context_builder.py :: build_context() L23
  |               dedupe >= .95 -> per-doc cap 3 -> token budget -> citation ids
  |  8 generate  clients/inference.py :: chat()     real cited answers via
  |              ollama; STILL degrades to answer:null if the LLM is down
  v
QueryResponse: answer, citations[], timings_ms, degraded[]

STREAMING VARIANT (stream: true + generate: true):
app/api/query.py :: query() L34 -> _stream_query() L55
  |  runs the SAME pipeline with generate=False (all stages above), then
  |  emits SSE (Server-Sent Events):
  |    event: citations   <- FIRST, so the client renders sources immediately
  |    event: token ...   <- clients/inference.py :: chat_stream() L102,
  |                          sync httpx stream bridged to async via to_thread
  |    event: done        <- usage totals
  |  LLM dies mid-stream? citations already delivered - degradation holds.
```

What to look for:
- The try/except around every stage: which exceptions DEGRADE (embedder,
  reranker, inference) vs which ERROR. This is the degradation contract in code.
- `build_context` L23: citation ids are born here and nowhere else - that is
  why every citation resolves.

Trace it yourself: open http://localhost:8000/ (the debug console), tick
"debug", run a query - the panel shows every intermediate stage this flow
just computed.

---

## FLOW 5 - BACKFILL: chunks -> H100 vectors (the batch path)

```
scripts/make_shards_local.py    Postgres -> data/shards/*.jsonl
  |  finds: dense IS NULL OR model mismatch   <- the version-stamp query
  v
modal volume put ...            laptop -> Modal volume
  v
modal_app/embed_h100.py :: Embedder.embed_shard()
  |  8 x H100 containers, model loaded once per container (@modal.enter)
  |  token-sorted batches, .done marker written LAST (crash-safe resume)
  v
modal volume get + scripts/load_vectors.py    vectors -> chunk_vectors + outbox
  v
FLOW 3 syncs them to Qdrant
```

What to look for: the `.done` marker pattern, and the lesson attached to it -
markers must be CLEARED when shards are re-exported (this bit us: 2 stale
markers silently skipped 4,000 chunks; a consistency count caught it).

---

## FLOW 6 - EVAL: retrieval -> a number you can defend

```
evaluation/run_eval.py       recall@K of HNSW vs chunks_flat (ANN error only)
evaluation/experiments/ef_sweep.py    the recall-vs-latency curve
evaluation/experiments/filter_selectivity.py   Experiment 5: does pre-filtering
                             hold recall as filters tighten? (yes: 1.0 at 0.002%)
evaluation/experiments/load_test.py   concurrency vs p95 - the HPA sizing number
evaluation/build_golden.py   content-anchored labels -> golden.jsonl
evaluation/run_golden.py     dense vs sparse vs hybrid vs full, hit@K + MRR
```

What to look for: build_golden's SPEC list - labels are (ticker, section,
must-contain phrases) resolved to chunk ids at run time, so re-chunking never
invalidates them.

---

## SUPPORT LAYER - read on demand, not cover-to-cover

| File | One line | Read when |
|---|---|---|
| app/config.py | every knob, from .env, one Settings object | a constant surprises you |
| app/persistence/db.py | asyncpg pool + tiny migration runner | before touching schema |
| app/main.py :: lifespan() L38 | startup order: pool -> migrations -> collections -> gauge task | debugging boot |
| app/observability/metrics.py | every Prometheus metric name | building dashboards |
| app/api/schemas.py | the public contract (mirrors openapi.json) | changing the API |
| migrations/*.sql | the schema, in order | understanding any table |
| app/observability/tracing.py | OTel setup; graceful no-op without endpoint | reading trace code |
| observability/ | prometheus+grafana+jaeger as code, alert rules | running the dashboards |
| deploy/k8s/ | the whole system as Kubernetes manifests (see its README) | the k8s deployment |

---

## Where CI/CD hooks in (next session's topic)

`.github/workflows/ci.yml` runs `pytest tests/unit` on every push - pure-math
tests (BM25, RRF, MMR, context builder, chunker packing, parser DP), no
services needed. `nightly-eval.yml` is scaffolded for scheduled FLOW 6 runs.
We will walk these when you are ready.

---

## Reading with production eyes - the self-test

You understand this codebase well enough to run it in production when you can
answer these from the code (not from the docs). Each names the file that holds
the answer - re-read it if you cannot.

1. A worker dies halfway through `insert_chunks`. What is in the database?
   (repository.py L232 - one transaction: NOTHING is. The retry redoes it all.)
2. Two workers claim jobs at the same instant. Can they get the same job?
   (repository.py L160 - SKIP LOCKED makes it impossible. Say why.)
3. Qdrant is down for an hour. What do users see? What does ingestion do?
   (pipeline.py: dense degrades, sparse... also Qdrant - so 503 only when BOTH
   paths fail; outbox rows just wait. No data loss either way.)
4. The LLM is down. Status code and response shape?
   (pipeline.py generate stage: 200, answer null, citations intact, degraded flag.)
5. A document is deleted while its chunks sit in a claimed outbox batch.
   Can deleted points resurrect in Qdrant forever?
   (outbox_worker.py drain(): upserts THEN tombstones each cycle - no.)
6. You change CHUNK_SIZE_TOKENS and re-ingest. What happens to old vectors?
   (chunker.py config_fingerprint L358: new ids; reconcile deletes stale rows
   + writes tombstones. Old id space cannot collide.)
7. The reaper resets a job whose worker is actually alive and finishes late.
   Who wins? (repository.finish_job L181: the worker_id fence - the late
   worker's finish is a no-op.)
8. What are the TWO metrics that catch silent death, and what do they look
   like when healthy? (metrics.py: rag_ingestion_jobs_pending,
   rag_outbox_pending_total - both near zero and not growing.)
9. A query with filters matching nothing: 200 or 503? Why must they differ?
   (pipeline.py: 200 with empty citations - outage and honest-empty are
   different answers.)
10. Where would you add a new pipeline stage so it is timed, traced, and
    degradable like the others? (pipeline.py: the `with t.stage(...)` +
    try/except pattern - copy it, never bypass it.)

When all ten are easy, you are ready to put this in front of users - and ready
for any interviewer who asks "what happens when X fails?"
