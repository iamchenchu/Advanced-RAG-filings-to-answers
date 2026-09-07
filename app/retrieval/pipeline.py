"""
retrieval/pipeline.py

The orchestrator. One query comes in; every stage of PRD 2.2 runs in order,
each stage is timed into rag_stage_duration_seconds AND into the timings_ms
block of the response, and each stage that fails DEGRADES instead of erroring:

    rewrite -> embed -> dense ‖ sparse -> RRF -> MMR -> rerank -> context -> generate

Degradation ladder (PRD section 5):
    embedder down   -> sparse-only, "dense_unavailable"
    qdrant down     -> retrieval_unavailable only if sparse also failed
    reranker down   -> fusion order, "rerank_skipped"
    inference down  -> answer=null, citations intact, "generation_unavailable"

A consumer never guesses: every shortcut taken is named in `degraded`.
"""

import asyncio
import time
import uuid
from contextlib import contextmanager

from app.clients.embedder import EmbedderUnavailable, get_embedder
from app.clients.inference import InferenceUnavailable, get_inference
from app.clients.reranker import RerankUnavailable, get_reranker
from app.config import get_settings
from app.observability.metrics import (CITATIONS_TOTAL, CONTEXT_TOKENS,
                                       QUERIES_TOTAL, RERANK_SCORE,
                                       RETRIEVED_DOCS, STAGE_DURATION)
from app.observability.tracing import span as trace_span
from app.retrieval.context_builder import build_context
from app.retrieval.dense import dense_search
from app.retrieval.fusion import rrf, weighted
from app.retrieval.mmr import mmr_select
from app.retrieval.sparse_search import sparse_search
from app.persistence import repository

SYSTEM_PROMPT = (
    "You are a financial research assistant answering questions about SEC filings.\n"
    "Answer ONLY from the numbered context passages below. Cite every claim with\n"
    "its passage number in square brackets, like [1] or [2][3]. If the context\n"
    "does not contain the answer, say so — never invent facts or citations."
)


class timings:
    """Collects stage -> milliseconds, and mirrors into Prometheus."""

    def __init__(self):
        self.data: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        # one wrapper, three outputs: timings_ms (API response), Prometheus
        # histogram (dashboards), and an OpenTelemetry span (trace waterfall).
        # A stage added through this manager is instrumented everywhere at once.
        t0 = time.perf_counter()
        with trace_span(f"rag.{name}"):
            try:
                yield
            finally:
                dt = time.perf_counter() - t0
                self.data[name] = round(dt * 1000, 1)
                STAGE_DURATION.labels(stage=name).observe(dt)


def generation_messages(query: str, context_text: str) -> list[dict]:
    """The exact prompt the non-streaming path uses - one builder, two paths,
    so streamed and non-streamed answers can never drift apart."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Context passages:\n\n{context_text}\n\nQuestion: {query}"},
    ]


async def run_query(query: str, top_k: int = 5, filters: dict | None = None,
                    generate: bool = True, include_debug: bool = False,
                    return_context: bool = False) -> dict:
    s = get_settings()
    t = timings()
    degraded: list[str] = []
    debug: dict = {"original_query": query}
    # root span: every stage span below nests under this one trace
    root = trace_span("rag.query", query_length=len(query), top_k=top_k,
                      generate=generate,
                      filters=",".join(sorted(filters)) if filters else "")
    root.__enter__()

    # ---- 1. rewrite (optional, LLM) ---------------------------------------
    search_query = query
    if s.query_rewrite_enabled and generate:
        with t.stage("rewrite"):
            try:
                out = await asyncio.to_thread(
                    get_inference().chat,
                    [{"role": "system",
                      "content": "Rewrite the user question as a standalone search query "
                                 "for a financial-filings search engine. Reply with the "
                                 "query only."},
                     {"role": "user", "content": query}],
                    max_tokens=100,
                )
                search_query = out["text"].strip() or query
            except InferenceUnavailable:
                degraded.append("rewrite_skipped")
    debug["rewritten_query"] = search_query

    # ---- 2. embed the query -----------------------------------------------
    # Blocking work (ONNX inference, sync qdrant client, cross-encoder, LLM
    # HTTP) runs via asyncio.to_thread so one slow query cannot freeze the
    # event loop for every other request (review #21).
    query_vector: list[float] | None = None
    dense_failed = False
    with t.stage("embed"):
        try:
            vec = await asyncio.to_thread(get_embedder().embed_one, search_query)
            query_vector = vec.tolist()
        except EmbedderUnavailable:
            degraded.append("dense_unavailable")
            dense_failed = True

    # ---- 3. dense + sparse search -----------------------------------------
    dense_hits: list[dict] = []
    if query_vector is not None:
        with t.stage("dense"):
            try:
                dense_hits = await asyncio.to_thread(
                    dense_search, query_vector, s.top_k_dense, filters)
            except Exception:                      # noqa: BLE001 — qdrant down
                dense_failed = True
                if "dense_unavailable" not in degraded:
                    degraded.append("dense_unavailable")

    sparse_hits: list[dict] = []
    sparse_failed = False
    if s.sparse_enabled:
        with t.stage("sparse"):
            try:
                sparse_hits = await sparse_search(search_query, s.top_k_sparse, filters)
            except Exception:                      # noqa: BLE001
                sparse_failed = True
                degraded.append("sparse_unavailable")

    # Outage and honest-empty are different answers (review #16): 503 only
    # when no retrieval PATH functioned. Zero matches from healthy paths (an
    # over-selective filter, an off-corpus query) is a valid empty result.
    if dense_failed and (sparse_failed or not s.sparse_enabled):
        QUERIES_TOTAL.labels(status="unavailable",
                             degraded=",".join(degraded) or "none").inc()
        root.__exit__(None, None, None)
        return {"error_type": "retrieval_unavailable", "degraded": degraded,
                "timings_ms": t.data}

    debug["dense_results"] = [{"chunk_id": str(h["chunk_id"]), "score": round(h["score"], 4)}
                              for h in dense_hits[:10]]
    debug["sparse_results"] = [{"chunk_id": str(h["chunk_id"]), "score": round(h["score"], 4)}
                               for h in sparse_hits[:10]]

    # ---- 4. fusion ---------------------------------------------------------
    with t.stage("fusion"):
        if s.fusion_strategy == "weighted":
            fused = weighted([dense_hits, sparse_hits],
                             [s.fusion_dense_weight, s.fusion_sparse_weight])
        else:
            fused = rrf([dense_hits, sparse_hits], k=s.rrf_k)
    debug["fusion_results"] = [{"chunk_id": str(h["chunk_id"]),
                                "fused_score": round(h["fused_score"], 5),
                                "source_ranks": h["source_ranks"]}
                               for h in fused[:10]]

    # ---- 5. MMR ------------------------------------------------------------
    with t.stage("mmr"):
        pool = mmr_select(fused, s.mmr_top_n, s.mmr_lambda) if s.mmr_enabled else fused[: s.mmr_top_n]
    debug["mmr_selected"] = [str(h["chunk_id"]) for h in pool]

    # ---- 5b. hydrate -------------------------------------------------------
    # Qdrant payloads are deliberately lean: chunk TEXT lives in Postgres, the
    # source of truth. Only the ~20 MMR survivors need their text (for rerank,
    # context, citations), so this is one cheap keyed lookup — and it doubles
    # as the citation-integrity guarantee: a point whose chunk row no longer
    # exists (index/DB skew) is dropped here, never cited.
    with t.stage("hydrate"):
        ids = [uuid.UUID(str(h["chunk_id"])) for h in pool]
        rows = await repository.chunks_by_ids(ids)
        hydrated = []
        for h in pool:
            row = rows.get(uuid.UUID(str(h["chunk_id"])))
            if row is None:
                continue
            h["text"] = row["text"]
            h["token_count"] = row["token_count"]
            h["document_name"] = row["document_name"]
            h["page_number"] = row["page_number"]
            h["section"] = row["section"]
            h["section_title"] = row["section_title"]
            hydrated.append(h)
        pool = hydrated

    # ---- 6. rerank ---------------------------------------------------------
    with t.stage("rerank"):
        if not pool:
            ranked = []            # nothing to rank is not a reranker failure
        else:
            try:
                scored = await asyncio.to_thread(
                    get_reranker().rerank, search_query, pool)
                ranked = scored[: max(top_k, 1)]
                for h in ranked:
                    RERANK_SCORE.observe(h["rerank_score"])
            except RerankUnavailable:
                degraded.append("rerank_skipped")
                ranked = pool[: max(top_k, 1)]
    debug["rerank_results"] = [{"chunk_id": str(h["chunk_id"]),
                                "rerank_score": round(h.get("rerank_score", 0.0), 4)}
                               for h in ranked]

    # ---- 7. context + citations -------------------------------------------
    with t.stage("context"):
        ctx = build_context(ranked)
    CONTEXT_TOKENS.observe(ctx["context_tokens"])
    RETRIEVED_DOCS.observe(ctx["n_documents"])
    CITATIONS_TOTAL.inc(len(ctx["citations"]))
    debug["context_chunk_ids"] = [c["chunk_id"] for c in ctx["citations"]]
    debug["context_dropped"] = ctx["dropped"]

    # ---- 8. generate -------------------------------------------------------
    answer = None
    usage = {"prompt_tokens": ctx["context_tokens"], "completion_tokens": 0}
    if generate and ctx["citations"]:          # no evidence -> no LLM call
        with t.stage("generate"):
            try:
                out = await asyncio.to_thread(
                    get_inference().chat,
                    generation_messages(query, ctx["context_text"]),
                )
                answer = out["text"]
                usage = out["usage"] or usage
            except InferenceUnavailable:
                degraded.append("generation_unavailable")

    t.data["total"] = round(sum(v for k, v in t.data.items() if k != "total"), 1)
    QUERIES_TOTAL.labels(status="ok", degraded=",".join(degraded) or "none").inc()
    root.__exit__(None, None, None)

    response = {
        **({"_context_text": ctx["context_text"]} if return_context else {}),
        "answer": answer,
        "citations": ctx["citations"],
        "usage": usage,
        "timings_ms": t.data,
        "degraded": degraded or None,
    }
    if include_debug:
        response["debug"] = debug
    return response
