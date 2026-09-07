"""
retrieval/dense.py

Dense (semantic) search against the HNSW collection.

ef_search rides along on EVERY query instead of being baked into the index —
that is what makes Experiment 1 (the ef_search sweep, PRD 7.4) a loop over a
parameter instead of a loop over index rebuilds.
"""

from qdrant_client import models

from app.clients.qdrant import DENSE_NAME, build_filter, get_client
from app.config import get_settings


def dense_search(query_vector: list[float], top_k: int,
                 filters: dict | None = None,
                 ef_search: int | None = None,
                 collection: str | None = None,
                 exact: bool = False) -> list[dict]:
    s = get_settings()
    client = get_client()
    result = client.query_points(
        collection_name=collection or s.qdrant_collection,
        query=query_vector,
        using=DENSE_NAME,
        limit=top_k,
        query_filter=build_filter(filters),          # PRE-filter, in-traversal
        search_params=models.SearchParams(
            hnsw_ef=ef_search or s.hnsw_ef_search,
            # With quantized vectors, HNSW walks the compressed index (fast)
            # but small quantization errors cap recall (~0.94 plateau measured).
            # Rescoring re-ranks the oversampled top candidates against the
            # full-precision vectors - recovering recall for ~1ms extra.
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=s.quantization_rescore,
                oversampling=2.0,
            ) if s.quantization != "none" else None,
            exact=exact,                             # True = flat ground truth
        ),
        with_payload=True,
        with_vectors=[DENSE_NAME],                   # MMR reuses these for free
    )
    return [
        {
            "chunk_id": p.id,
            "score": p.score,
            "source": "dense",
            "vector": p.vector[DENSE_NAME] if isinstance(p.vector, dict) else p.vector,
            **(p.payload or {}),
        }
        for p in result.points
    ]
