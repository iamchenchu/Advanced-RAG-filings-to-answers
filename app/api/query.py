"""
api/query.py — POST /api/v1/query, the endpoint Multi-Agent consumes.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.schemas import QueryRequest, QueryResponse
from app.retrieval.pipeline import run_query

router = APIRouter()


# NOTE: no response_model_exclude_none — the degradation contract REQUIRES
# "answer": null to appear (with citations intact) when generation is down.
@router.post(
    "/query", response_model=QueryResponse,
    responses={   # the published contract must show the error envelopes too
        400: {"description": "invalid_filter — unknown or malformed filter key",
              "content": {"application/json": {"example": {
                  "error": {"type": "invalid_filter",
                            "message": "filters.bogus: Extra inputs are not permitted"}}}}},
        503: {"description": "retrieval_unavailable — no retrieval path functioned",
              "content": {"application/json": {"example": {
                  "error": {"type": "retrieval_unavailable",
                            "message": "no retrieval path available",
                            "degraded": ["dense_unavailable", "sparse_unavailable"]}}}}},
    })
async def query(req: QueryRequest):
    filters = req.filters.model_dump(exclude_none=True) if req.filters else None
    try:
        result = await run_query(
            query=req.query, top_k=req.top_k, filters=filters,
            generate=req.generate, include_debug=req.include_debug,
        )
    except ValueError as exc:                       # unknown filter keys
        return JSONResponse(status_code=400, content={
            "error": {"type": "invalid_filter", "message": str(exc)}})

    if result.get("error_type"):                    # both retrieval paths dead
        return JSONResponse(status_code=503, content={
            "error": {"type": result["error_type"],
                      "message": "no retrieval path available",
                      "degraded": result.get("degraded")}})
    return result
