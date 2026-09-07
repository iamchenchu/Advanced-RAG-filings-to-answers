"""
clients/qdrant.py

The only file that talks to Qdrant.

Two collections, same points (PRD 3.3 / 7.3):

  chunks       HNSW + configurable quantization. What production queries hit.
  chunks_flat  hnsw disabled, searched with exact=True. Ground truth: Recall@K
               of the HNSW index is measured against THIS, and is undefined
               without it.

Point id = chunk_id (the deterministic uuid5 from chunker.py), which makes
every upsert idempotent: re-syncing the same chunk overwrites the same point.
"""

import uuid

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.observability.metrics import QDRANT_ERRORS

SPARSE_NAME = "bm25"
DENSE_NAME = "dense"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None,
                               timeout=s.qdrant_timeout_s)
    return _client


def _quantization_config(kind: str):
    if kind == "scalar":
        return models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8, always_ram=True))
    if kind == "product":
        return models.ProductQuantization(product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X16, always_ram=True))
    if kind == "binary":
        return models.BinaryQuantization(binary=models.BinaryQuantizationConfig(
            always_ram=True))
    return None


def ensure_collections() -> None:
    """Create both collections if missing. Safe to call at every startup."""
    s = get_settings()
    client = get_client()

    def _make(name: str, hnsw_m: int) -> None:
        if client.collection_exists(name):
            # a collection built for a different embedding dim would accept
            # writes and then fail (or worse, mis-search) — fail loudly instead
            info = client.get_collection(name)
            have = info.config.params.vectors[DENSE_NAME].size
            if have != s.embedding_dim:
                raise RuntimeError(
                    f"collection {name} holds {have}-dim vectors but "
                    f"EMBEDDING_DIM={s.embedding_dim}; drop and re-sync "
                    f"(outbox --resync) after changing embedding models")
            return
        client.create_collection(
            collection_name=name,
            vectors_config={DENSE_NAME: models.VectorParams(
                size=s.embedding_dim,
                distance=models.Distance.COSINE,
                on_disk=True,               # 1M+ x 1024-dim x 2 collections
                                            # cannot live in the docker VM's RAM
                hnsw_config=models.HnswConfigDiff(
                    m=hnsw_m,
                    ef_construct=s.hnsw_ef_construct,
                    full_scan_threshold=s.hnsw_full_scan_threshold,
                    on_disk=True,
                ),
            )},
            sparse_vectors_config={SPARSE_NAME: models.SparseVectorParams(
                # weights are hand-computed BM25 (PRD 3.5: writing it is the
                # point) — no server-side IDF modifier.
                modifier=models.Modifier.NONE,
                index=models.SparseIndexParams(on_disk=True),
            )},
            quantization_config=_quantization_config(s.quantization) if hnsw_m else None,
        )
        # payload indexes: pre-filtering happens during graph traversal, and
        # an indexed field is what keeps a selective filter from crawling
        for field, ftype in [("tenant", models.PayloadSchemaType.KEYWORD),
                             ("ticker", models.PayloadSchemaType.KEYWORD),
                             ("section", models.PayloadSchemaType.KEYWORD),
                             ("fiscal_year", models.PayloadSchemaType.KEYWORD),
                             ("is_narrative", models.PayloadSchemaType.BOOL),
                             ("document_id", models.PayloadSchemaType.KEYWORD),
                             ("filing_date", models.PayloadSchemaType.DATETIME)]:
            client.create_payload_index(name, field_name=field, field_schema=ftype)

    _make(s.qdrant_collection, s.hnsw_m)
    _make(s.qdrant_flat_collection, 0)          # m=0 disables the HNSW graph


def upsert_points(rows: list[dict], bm25_weights: list[dict[int, float]]) -> None:
    """One batch of outbox rows -> points in BOTH collections."""
    s = get_settings()
    client = get_client()
    points = []
    for row, weights in zip(rows, bm25_weights):
        meta = row.get("doc_meta") or {}
        points.append(models.PointStruct(
            id=str(row["chunk_id"]),
            vector={
                DENSE_NAME: list(row["dense"]),
                SPARSE_NAME: models.SparseVector(
                    indices=list(weights.keys()), values=list(weights.values())),
            },
            payload={
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "document_name": row["document_name"],
                "chunk_index": row["chunk_index"],
                "section": row["section"],
                "section_title": row["section_title"],
                "is_narrative": row["is_narrative"],
                "token_count": row["token_count"],
                "tenant": row["tenant"],
                "region": row["region"],
                "ticker": meta.get("ticker"),
                "company": meta.get("company"),
                # str() coercion: an integer year in documents.meta would
                # never match the keyword payload index or a string filter
                "fiscal_year": str(meta["fiscal_year"]) if meta.get("fiscal_year") is not None else None,
                "filing_date": str(meta["filing_date"]) if meta.get("filing_date") is not None else None,
                "tags": list(row.get("doc_tags") or []),
                "source_uri": row.get("gcs_uri"),
                "embedding_model": row["embedding_model"],
            },
        ))
    try:
        for name in (s.qdrant_collection, s.qdrant_flat_collection):
            client.upsert(collection_name=name, points=points, wait=True)
    except Exception:
        QDRANT_ERRORS.inc()
        raise


def delete_points(chunk_ids: list[uuid.UUID]) -> None:
    s = get_settings()
    client = get_client()
    ids = [str(c) for c in chunk_ids]
    for name in (s.qdrant_collection, s.qdrant_flat_collection):
        if client.collection_exists(name):
            client.delete(collection_name=name,
                          points_selector=models.PointIdsList(points=ids), wait=True)


def build_filter(filters: dict | None) -> models.Filter | None:
    """API filter dict -> Qdrant PRE-filter (applied during traversal).

    Post-filtering is a correctness bug, not a performance choice: filter
    top-100 AFTER search and 97 failures leave 3 results. Pre-filtering
    returns the full top-K that satisfies the filter (PRD 3.6).
    """
    if not filters:
        return None
    known = {"tenant", "ticker", "section", "item", "fiscal_year", "is_narrative",
             "document_id", "date_after", "date_before", "tags"}
    unknown = set(filters) - known
    if unknown:
        raise ValueError(f"unknown filter keys: {sorted(unknown)}")

    must: list[models.Condition] = []
    key_map = {"item": "section"}               # API says item, payload says section
    for key in ("tenant", "ticker", "section", "item", "fiscal_year", "document_id"):
        if key in filters and filters[key] is not None:
            field = key_map.get(key, key)
            value = filters[key]
            if isinstance(value, list):
                must.append(models.FieldCondition(key=field, match=models.MatchAny(any=value)))
            else:
                must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if filters.get("tags"):
        # a chunk matches if its document carries ANY of the requested tags
        must.append(models.FieldCondition(key="tags",
                    match=models.MatchAny(any=list(filters["tags"]))))
    if "is_narrative" in filters and filters["is_narrative"] is not None:
        must.append(models.FieldCondition(key="is_narrative",
                    match=models.MatchValue(value=bool(filters["is_narrative"]))))
    if filters.get("date_after") or filters.get("date_before"):
        must.append(models.FieldCondition(key="filing_date", range=models.DatetimeRange(
            gte=filters.get("date_after"), lte=filters.get("date_before"))))
    return models.Filter(must=must) if must else None
