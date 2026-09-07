"""
main.py — FastAPI assembly.

    uvicorn app.main:app --port 8000

Startup: DB pool, migrations, Qdrant collections (all idempotent), plus the
gauge-refresher task that keeps rag_ingestion_jobs_pending and
rag_outbox_pending_total honest — the two metrics that catch silent death.
"""

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.api import documents, evaluations, health, query
from app.config import get_settings
from app.observability.metrics import INGESTION_PENDING, OUTBOX_PENDING
from app.persistence import repository
from app.persistence.db import close_pool, get_pool, run_migrations

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


async def _refresh_gauges() -> None:
    while True:
        with contextlib.suppress(Exception):
            depths = await repository.queue_depths()
            INGESTION_PENDING.set(depths["jobs_pending"])
            OUTBOX_PENDING.set(depths["outbox_pending"])
        await asyncio.sleep(15)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()
    await run_migrations()
    with contextlib.suppress(Exception):        # qdrant down: API still boots,
        from app.clients.qdrant import ensure_collections
        ensure_collections()                    # degradation handles the rest
    task = asyncio.create_task(_refresh_gauges())
    yield
    task.cancel()
    await close_pool()


settings = get_settings()
app = FastAPI(
    title="Advanced-RAG-System",
    version="1.0.0",
    lifespan=lifespan,
    description="Hybrid retrieval over SEC filings: dense HNSW + hand-written BM25, "
                "RRF fusion, MMR, cross-encoder rerank, citation-aware generation.",
)

app.include_router(query.router, prefix=settings.api_base_path, tags=["query"])
app.include_router(documents.router, prefix=settings.api_base_path, tags=["documents"])
app.include_router(evaluations.router, prefix=settings.api_base_path, tags=["evaluations"])
app.include_router(health.router, tags=["ops"])


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Contract error envelope. A bad `filters` object maps to the documented
    invalid_filter type; any other validation failure stays a generic 422."""
    errors = exc.errors()
    if any("filters" in e.get("loc", ()) for e in errors):
        return JSONResponse(status_code=400, content={
            "error": {"type": "invalid_filter",
                      "message": "; ".join(f"{'.'.join(str(l) for l in e['loc'][1:])}: {e['msg']}"
                                            for e in errors)}})
    return JSONResponse(status_code=422, content={
        "error": {"type": "validation_error", "message": str(errors)}})


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(FRONTEND)
