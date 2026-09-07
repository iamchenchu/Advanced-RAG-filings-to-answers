"""
api/evaluations.py — trigger and read evaluation runs.

The math lives in evaluation/run_eval.py (also runnable as a CLI); this
endpoint just runs it in the background and stores results in rag_db.
"""

import uuid

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

from app.api.schemas import EvalRequest
from app.persistence import repository

router = APIRouter()


@router.post("/evaluations", status_code=202)
async def trigger_eval(req: EvalRequest, background: BackgroundTasks):
    from evaluation.run_eval import evaluate_recall     # local import: avoids cycle
    run_id = await repository.create_eval_run(
        req.config_name, {"k_values": req.k_values, "ef_search": req.ef_search})
    background.add_task(evaluate_recall, run_id, req.valid_k, req.ef_search,
                        req.config_name)
    return {"run_id": str(run_id), "status": "running"}


@router.get("/evaluations/{run_id}")
async def get_eval(run_id: uuid.UUID):
    run = await repository.get_eval_run(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={
            "error": {"type": "not_found", "message": f"evaluation run {run_id}"}})
    run["id"] = str(run["id"])
    return run
