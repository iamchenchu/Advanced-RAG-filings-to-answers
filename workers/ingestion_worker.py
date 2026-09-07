"""
workers/ingestion_worker.py

The pipeline half that turns a registered document into embedded chunks:

    claim job (SKIP LOCKED)
      -> load source (GCS / inline text)
      -> parse (10-K item sections)
      -> chunk (token windows, deterministic ids)
      -> dense embed (batched)  +  BM25 term counts
      -> ONE transaction: chunks + vectors + stats + outbox rows
      -> mark job done

Crash-safety story, because this is the part that must survive the 5,000-
filing backfill: the job row stays 'processing' if the worker dies; the
reaper resets it; the retry re-runs everything, and ON CONFLICT (id) DO
NOTHING on deterministic chunk ids makes the re-run insert only what is
missing. Kill this process at any line and no data is lost or doubled.

Run:  python workers/ingestion_worker.py           # drain queue, then poll
      python workers/ingestion_worker.py --once    # drain queue, then exit
"""

import argparse
import asyncio
import os
import socket
import sys
import traceback
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.embedder import get_embedder                     # noqa: E402
from app.config import get_settings                               # noqa: E402
from app.ingestion.chunker import Tokenizer, chunk_sections       # noqa: E402
from app.ingestion.parser import (html_to_text, load_html,        # noqa: E402
                                  split_into_items)
from app.observability.metrics import INGESTION_FAILED            # noqa: E402
from app.persistence import repository                            # noqa: E402
from app.persistence.db import close_pool, run_migrations         # noqa: E402
from app.retrieval.sparse import encode_document                  # noqa: E402

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


def process_document(doc: dict, tok: Tokenizer) -> list[dict]:
    """Everything CPU-bound: parse -> chunk -> embed -> sparse-encode.
    Pure function of the document; safe to re-run any number of times."""
    meta = doc.get("meta") or {}

    if meta.get("inline_text"):
        sections = {"0": meta["inline_text"]}          # ad-hoc docs: one section
    else:
        html = load_html(doc["gcs_uri"])
        text = html_to_text(html)
        sections = split_into_items(text)
        if not sections:                               # not a 10-K: whole doc
            sections = {"0": text}

    chunk_meta = {
        "source_uri": doc.get("gcs_uri") or f"inline:{doc['id']}",
        "tenant": doc["tenant"],
        "ticker": meta.get("ticker"),
        "company": meta.get("company"),
        "cik": meta.get("cik"),
        "filing_date": meta.get("filing_date"),
        "fiscal_year": meta.get("fiscal_year"),
        "form_type": meta.get("form_type", "10-K"),
    }
    chunks, _stats = chunk_sections(sections, chunk_meta, tok)
    if not chunks:
        return []

    dense = get_embedder().embed([c.embed_text for c in chunks])
    if dense is None:                       # defer backend: H100 fills these in
        dense = [None] * len(chunks)

    rows: list[dict] = []
    for chunk, vec in zip(chunks, dense):
        tf, terms, doc_len = encode_document(chunk.text)
        rows.append({
            "id": uuid.UUID(chunk.chunk_id),
            "chunk_index": chunk.chunk_index,
            "section_chunk_index": chunk.section_chunk_index,
            "text": chunk.text,
            "context_header": chunk.context_header,
            "section": chunk.item,
            "section_title": chunk.item_title,
            "is_narrative": chunk.is_narrative,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "content_sha256": chunk.content_sha256,
            "tenant": chunk.tenant,
            "region": doc.get("region", "us"),
            "embedding_model": get_settings().embedding_model,
            "chunk_config": chunk.chunk_config,
            "dense": [float(x) for x in vec] if vec is not None else None,
            "sparse_tf": tf,
            "sparse_terms": terms,
            "doc_len": doc_len,
        })
    return rows


async def work_loop(once: bool) -> None:
    s = get_settings()
    await run_migrations()
    tok = Tokenizer(s.embedding_model)              # measured with the real model
    print(f"[{WORKER_ID}] ingestion worker up (backend={s.embedding_backend})")

    while True:
        job = await repository.claim_job(WORKER_ID, s.ingestion_max_attempts)
        if job is None:
            if once:
                print(f"[{WORKER_ID}] queue empty, exiting")
                return
            await asyncio.sleep(2)
            continue

        doc = await repository.get_document(job["document_id"])
        if doc is None:
            # document deleted between claim and lookup (review #7) — the job
            # is moot, not a crash
            await repository.finish_job(job["id"], "done",
                                        "document deleted before processing",
                                        worker_id=WORKER_ID)
            continue
        print(f"[{WORKER_ID}] job {job['id']}: {doc['name']} (attempt {job['attempts']})")
        try:
            await repository.set_document_status(doc["id"], "processing")
            # CPU-heavy work off the event loop so /ready stays responsive
            rows = await asyncio.to_thread(process_document, doc, tok)
            outcome = await repository.insert_chunks(doc["id"], rows)
            # fenced: if the reaper reassigned this job while we worked, the
            # other worker's run owns the finish — ours is a no-op
            owned = await repository.finish_job(job["id"], "done",
                                                worker_id=WORKER_ID)
            if owned:
                await repository.set_document_status(doc["id"], "ready")
            print(f"[{WORKER_ID}] job {job['id']}: {outcome['inserted']} inserted, "
                  f"{outcome['updated']} updated, {outcome['unchanged']} unchanged, "
                  f"{outcome['deleted']} stale removed"
                  + ("" if owned else "  [job was reaped mid-run; status left alone]"))
        except Exception as exc:                    # noqa: BLE001
            INGESTION_FAILED.inc()
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            final = job["attempts"] >= s.ingestion_max_attempts
            owned = await repository.finish_job(
                job["id"], "dead" if final else "pending", error,
                worker_id=WORKER_ID)
            if owned:
                await repository.set_document_status(
                    doc["id"], "failed" if final else "pending", error)


async def _run(once: bool) -> None:
    try:
        await work_loop(once)
    finally:
        await close_pool()          # same event loop as the pool — clean exit


def _maybe_serve_metrics() -> None:
    """Workers mutate their own metric registry (INGESTION_FAILED etc), which
    the API's /metrics can never see (separate process). Set METRICS_PORT to
    expose this worker's registry for Prometheus to scrape."""
    import os
    port = os.getenv("METRICS_PORT")
    if port:
        from prometheus_client import start_http_server
        start_http_server(int(port))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="drain the queue, then exit")
    args = ap.parse_args()
    _maybe_serve_metrics()
    asyncio.run(_run(args.once))


if __name__ == "__main__":
    main()
