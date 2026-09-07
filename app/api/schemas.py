"""
api/schemas.py

Pydantic models for the public contract (PRD section 4). These models ARE the
openapi.json that Multi-Agent generates its client from — change them and you
are changing the published rag-v1 contract.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    # extra="forbid": an unknown filter key is a 400 invalid_filter, never a
    # silent no-op — silently ignoring a filter the consumer asked for would
    # return unfiltered results that LOOK filtered.
    model_config = {"extra": "forbid"}

    tenant: str | None = None
    ticker: str | list[str] | None = None
    item: str | list[str] | None = None
    fiscal_year: str | list[str] | None = None
    is_narrative: bool | None = None
    document_id: str | None = None
    date_after: str | None = None
    date_before: str | None = None
    tags: list[str] | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: QueryFilters | None = None
    stream: bool = False
    include_debug: bool = False
    # THE key contract decision: generate=false returns passages with no LLM
    # call. The Multi-Agent research agent usually wants passages, not prose.
    generate: bool = True


class Citation(BaseModel):
    id: int
    document_id: str
    document_name: str | None
    page_number: int | None = None
    section: str | None = None
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[Citation]
    usage: dict[str, Any]
    timings_ms: dict[str, float]
    degraded: list[str] | None = None
    debug: dict[str, Any] | None = None


class DocumentIn(BaseModel):
    """Register a document already sitting in GCS (the EDGAR flow), or send
    inline text for small ad-hoc documents."""
    name: str
    gcs_uri: str | None = None
    text: str | None = None
    source_type: str = "html"
    tenant: str | None = None
    region: str | None = None
    tags: list[str] = []
    meta: dict[str, Any] = {}     # ticker, company, cik, fiscal_year, filing_date


class DocumentAccepted(BaseModel):
    document_id: uuid.UUID
    job_id: int
    status: str = "pending"


class EvalRequest(BaseModel):
    config_name: str = "default"
    k_values: list[int] = Field(default=[1, 5, 10])
    ef_search: int | None = Field(default=None, ge=4, le=4096)

    @property
    def valid_k(self) -> list[int]:
        return [k for k in self.k_values if k >= 1] or [10]


class ErrorBody(BaseModel):
    type: str
    message: str
