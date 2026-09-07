"""
api/documents.py — upload / list / status / delete.

POST answers 202 immediately: ingesting a document is minutes of work and an
HTTP request must never block on it. The ingestion worker picks the job up
from Postgres; progress is visible at GET /documents/{id}/status.
"""

import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.schemas import DocumentAccepted, DocumentIn
from app.config import get_settings
from app.persistence import repository

router = APIRouter()


@router.post("/documents", response_model=DocumentAccepted, status_code=202)
async def upload_document(doc: DocumentIn):
    s = get_settings()
    if not doc.gcs_uri and not doc.text:
        return JSONResponse(status_code=400, content={
            "error": {"type": "validation_error",
                      "message": "one of gcs_uri or text is required"}})
    meta = dict(doc.meta)
    if doc.text is not None:
        meta["inline_text"] = doc.text          # small docs ride along in meta
    row = await repository.create_document(
        name=doc.name, source_type=doc.source_type, gcs_uri=doc.gcs_uri,
        tenant=doc.tenant or s.default_tenant, region=doc.region or s.default_region,
        tags=doc.tags, meta=meta,
    )
    return DocumentAccepted(document_id=row["id"], job_id=row["job_id"])


@router.get("/documents")
async def list_documents(tenant: str | None = None, limit: int = 100):
    docs = await repository.list_documents(tenant, limit)
    for d in docs:
        d["id"] = str(d["id"])
        d.pop("meta", None)                      # can hold inline_text; keep list light
    return {"documents": docs}


@router.get("/documents/{document_id}/status")
async def get_status(document_id: uuid.UUID):
    status = await repository.document_status(document_id)
    if status is None:
        return JSONResponse(status_code=404, content={
            "error": {"type": "document_not_found", "message": str(document_id)}})
    status["id"] = str(status["id"])
    return status


@router.delete("/documents/{document_id}")
async def delete_document(document_id: uuid.UUID):
    doc = await repository.get_document(document_id)
    if doc is None:
        return JSONResponse(status_code=404, content={
            "error": {"type": "document_not_found", "message": str(document_id)}})
    chunk_ids = await repository.delete_document(document_id)
    # Qdrant purge happens via tombstones written in the SAME transaction as
    # the row deletes and drained by the outbox worker — crash-safe where a
    # direct synchronous purge was not (review findings #2/#20). Until the
    # worker drains them, stale points are dropped at the hydrate stage, so
    # deleted content is never CITED even in the sync gap.
    return {"deleted": str(document_id), "chunks_removed": len(chunk_ids),
            "vector_purge": "queued"}
