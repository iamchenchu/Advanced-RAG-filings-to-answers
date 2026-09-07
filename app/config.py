"""
config.py

One Settings object, loaded from .env, injected everywhere.

Why pydantic-settings instead of os.environ scattered through the code: every
retrieval knob in this system is an EXPERIMENT PARAMETER (PRD 3.2, 7.4). An
experiment is "change one env var, re-run, compare numbers" — that only works
if there is exactly one place where configuration enters the process.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # service
    env: str = "local"
    log_level: str = "info"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_path: str = "/api/v1"

    # postgres
    database_url: str = "postgresql://rag:changeme@localhost:5432/rag_db"
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "chunks"
    qdrant_flat_collection: str = "chunks_flat"
    qdrant_timeout_s: int = 10
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    hnsw_full_scan_threshold: int = 10000
    hnsw_ef_search: int = 128
    quantization: str = "none"                     # none|scalar|product|binary
    quantization_always_ram: bool = True
    quantization_rescore: bool = True              # re-check top hits at full precision
    quantization_rescore_limit: int = 100

    # embeddings
    embedding_backend: str = "fastembed"           # tei|fastembed|hash
    embedding_url: str = "http://localhost:8081"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    embedding_timeout_s: int = 30
    embedding_max_retries: int = 3

    # reranker
    reranker_enabled: bool = True
    reranker_backend: str = "fastembed"            # tei|fastembed|none
    reranker_url: str = "http://localhost:8082"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 100
    reranker_return_n: int = 5
    reranker_timeout_s: int = 10

    # inference
    inference_url: str = "http://localhost:8080/v1"
    inference_model: str = "qwen3.6-35b-a3b"
    inference_api_key: str = ""
    inference_connect_timeout_s: float = 2
    inference_first_token_timeout_s: float = 30
    inference_total_timeout_s: float = 300
    inference_max_retries: int = 1

    # retrieval pipeline
    query_rewrite_enabled: bool = False
    top_k_dense: int = 100
    top_k_sparse: int = 100
    sparse_enabled: bool = True
    fusion_strategy: str = "rrf"                   # rrf|weighted
    rrf_k: int = 60
    fusion_dense_weight: float = 0.5
    fusion_sparse_weight: float = 0.5
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7
    mmr_top_n: int = 20

    # BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # context builder
    max_context_tokens: int = 100000
    context_reserve_tokens: int = 2000
    dedupe_threshold: float = 0.95
    max_chunks_per_document: int = 3

    # ingestion
    ingestion_worker_concurrency: int = 2
    ingestion_max_attempts: int = 3
    ingestion_stuck_job_timeout_min: int = 30
    outbox_poll_interval_s: float = 2
    outbox_batch_size: int = 500

    # storage
    gcs_bucket: str = ""
    gcs_prefix: str = "filings/"
    sec_user_agent: str = ""

    # tenancy
    default_tenant: str = "default"
    default_region: str = "us"

    # auth / rate limit
    auth_enabled: bool = False
    rate_limit_enabled: bool = False
    rate_limit_queries_per_minute: int = 120

    metrics_enabled: bool = True
    otel_exporter_otlp_endpoint: str = ""      # empty = tracing off
    otel_service_name: str = "advanced-rag-system"


@lru_cache
def get_settings() -> Settings:
    return Settings()
