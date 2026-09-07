"""
metrics.py

Every metric named in PRD section 8, defined once.

The two that matter most are the SILENT-failure detectors:
  rag_ingestion_jobs_pending  — growing means workers are dead
  rag_outbox_pending_total    — growing means Qdrant sync is dead and
                                retrieval is quietly going stale
Neither produces an exception anywhere; only a gauge catches them.
"""

from prometheus_client import Counter, Gauge, Histogram

STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds",
    "Duration of each retrieval pipeline stage",
    ["stage"],   # rewrite|embed|dense|sparse|fusion|mmr|rerank|context|generate
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

QUERIES_TOTAL = Counter(
    "rag_queries_total", "Queries served", ["status", "degraded"]
)

CONTEXT_TOKENS = Histogram(
    "rag_context_tokens", "Tokens sent to the LLM as context",
    buckets=(256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
)

RETRIEVED_DOCS = Histogram(
    "rag_retrieved_docs", "Distinct documents in the final context",
    buckets=(1, 2, 3, 5, 8, 13, 21),
)

RERANK_SCORE = Histogram(
    "rag_rerank_score", "Reranker score distribution (watch for index drift)",
    buckets=(-10, -5, -2, -1, 0, 1, 2, 5, 10),
)

CITATIONS_TOTAL = Counter("rag_citations_total", "Citations returned")

RECALL_AT_10 = Gauge("rag_recall_at_10", "Recall@10 vs flat ground truth (last eval run)")

INGESTION_PENDING = Gauge("rag_ingestion_jobs_pending", "Ingestion jobs waiting")
INGESTION_FAILED = Counter("rag_ingestion_jobs_failed_total", "Ingestion jobs failed")
OUTBOX_PENDING = Gauge("rag_outbox_pending_total", "Outbox rows not yet synced to Qdrant")
QDRANT_ERRORS = Counter("rag_qdrant_errors_total", "Qdrant call failures")
