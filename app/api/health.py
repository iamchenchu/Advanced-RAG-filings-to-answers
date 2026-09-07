"""
api/health.py — /health, /ready, /metrics.

/health  liveness: the process is up. Always 200.
/ready   readiness: can this pod serve? rag_db unreachable -> 503 -> the
         Service stops routing here (PRD 5: fail fast on the source of truth;
         everything else degrades, the database does not).
/metrics Prometheus. The queue-depth gauges are refreshed by a background
         task in main.py, because silent failures only show up in gauges.
"""

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.persistence.db import get_pool

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
    except Exception as exc:                    # noqa: BLE001
        return JSONResponse(status_code=503,
                            content={"status": "not_ready", "reason": str(exc)})
    return {"status": "ready"}


@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
