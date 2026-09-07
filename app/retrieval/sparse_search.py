"""
retrieval/sparse_search.py

Query side of BM25: build idf weights from rag_db statistics, run the sparse
search in Qdrant. Kept separate from sparse.py so the math module stays pure
(unit-testable with no database or Qdrant running).
"""

import asyncio

from qdrant_client import models

from app.clients.qdrant import SPARSE_NAME, build_filter, get_client
from app.config import get_settings
from app.persistence import repository
from app.retrieval.sparse import analyze, query_weights, term_id


async def sparse_search(query: str, top_k: int,
                        filters: dict | None = None) -> list[dict]:
    """BM25 search. Returns [] when the query shares no terms with the corpus
    — that is an honest empty result, not an error."""
    term_ids = [term_id(t) for t in set(analyze(query))]
    if not term_ids:
        return []
    dfs = await repository.term_dfs(term_ids)
    stats = await repository.corpus_stats()
    weights = query_weights(query, dfs, stats["n_chunks"])
    if not weights:
        return []

    s = get_settings()
    client = get_client()
    result = await asyncio.to_thread(
        lambda: client.query_points(
        collection_name=s.qdrant_collection,
        query=models.SparseVector(indices=list(weights.keys()),
                                  values=list(weights.values())),
        using=SPARSE_NAME,
        limit=top_k,
        query_filter=build_filter(filters),
        with_payload=True,
        with_vectors=["dense"],          # so sparse-only hits still join MMR
    ))
    return [
        {
            "chunk_id": p.id,
            "score": p.score,
            "source": "sparse",
            "vector": (p.vector or {}).get("dense") if isinstance(p.vector, dict) else None,
            **(p.payload or {}),
        }
        for p in result.points
    ]
