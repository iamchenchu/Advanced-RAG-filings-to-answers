"""
api/query.py — POST /api/v1/query, the endpoint Multi-Agent consumes.
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
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
    if req.stream and req.generate:
        return await _stream_query(req)

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


async def _stream_query(req: QueryRequest):
    """stream: true - Server-Sent Events (SSE).

    Event order is the contract:
      1. `citations`  - the full retrieval result (ids resolve immediately, so
                        a consumer can render sources before any prose exists)
      2. `delta`      - answer text fragments as the LLM emits them
      3. `done`       - degraded flags; or `generation_unavailable` if the LLM
                        failed before/while streaming (citations still stand)
    """
    from app.clients.inference import InferenceUnavailable, get_inference
    from app.retrieval.pipeline import generation_messages, run_query

    filters = req.filters.model_dump(exclude_none=True) if req.filters else None
    result = await run_query(
        query=req.query, top_k=req.top_k, filters=filters,
        generate=False, include_debug=req.include_debug, return_context=True,
    )
    if result.get("error_type"):        # retrieval itself unavailable: plain JSON error
        return JSONResponse(status_code=503, content={
            "error": {"type": result["error_type"],
                      "message": "no retrieval path available",
                      "degraded": result.get("degraded", [])}})

    context_text = result.pop("_context_text", "")

    async def events():
        payload = {k: v for k, v in result.items() if k != "answer"}
        yield f"event: citations\ndata: {json.dumps(payload)}\n\n"

        degraded = list(result.get("degraded") or [])
        if result["citations"]:
            messages = generation_messages(req.query, context_text)
            gen = get_inference().chat_stream(messages)
            sentinel = object()
            try:
                while True:
                    # the client generator is synchronous (httpx stream);
                    # step it off the event loop one chunk at a time
                    chunk = await asyncio.to_thread(next, gen, sentinel)
                    if chunk is sentinel:
                        break
                    yield f"event: delta\ndata: {json.dumps({'text': chunk})}\n\n"
            except InferenceUnavailable:
                degraded.append("generation_unavailable")
        yield f"event: done\ndata: {json.dumps({'degraded': degraded})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
