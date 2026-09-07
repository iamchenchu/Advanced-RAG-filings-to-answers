"""
modal_app/embed_backfill.py — the PRODUCTION embedding path: BGE-M3 on Modal
H100s, for the 5,000-filing / ~1.5M-chunk backfill.

Local dev embeds in-process (fastembed, small model). That is fine for 118
chunks and useless for 1.5M: bge-m3 on CPU is ~20 chunks/s single-threaded —
about 20 hours of laptop time — while one H100 does thousands per second.

The design is the DATA-PIPELINE shape, resumable at every boundary:

    scripts/make_shards.py        Postgres -> GCS   chunks needing vectors,
                                                    JSONL shards of ~2k chunks
    THIS FILE (modal deploy/run)  GCS -> H100 -> GCS  one container per shard,
                                                    writes <shard>.vectors.jsonl
                                                    + a .done marker
    scripts/load_embeddings.py    GCS -> Postgres   upsert chunk_vectors,
                                                    enqueue outbox rows
    workers/outbox_worker.py      Postgres -> Qdrant  same as always

If a GPU dies at shard 350/750: re-run. Done-markers skip completed shards;
the loader's upsert is idempotent; the outbox dedupes pending rows. Nothing
restarts from zero — the property the whole system is designed around.

Cost sanity: 1.5M chunks x 512 tokens ≈ 750M tokens. An H100 runs bge-m3
inference at roughly 5-10k chunks/min with token-sorted batches, so the whole
backfill is a few GPU-hours across N parallel containers — not days.

Setup (one time):
    pip install modal && modal setup
    modal secret create gcp-credentials GOOGLE_APPLICATION_CREDENTIALS_JSON=<service-account-json>

Run:
    modal run modal_app/embed_backfill.py --shard-prefix gs://<bucket>/embed_jobs/2026-08-18
"""

import json
import subprocess

import modal

app = modal.App("rag-embed-backfill")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "FlagEmbedding==1.3.5",          # canonical BGEM3FlagModel
        "torch",                          # CUDA wheels resolve inside the image
        "google-cloud-storage",
        "numpy",
    )
)

MODEL = "BAAI/bge-m3"
BATCH_MAX_TOKENS = 16384        # dynamic token-based batches (TEI-style):
                                # batch by TOKENS, not chunk count, so short
                                # chunks pack densely and long ones don't OOM


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    retries=2,                              # transient CUDA/network failures
    secrets=[modal.Secret.from_name("gcp-credentials")],
    max_containers=8,                       # 8 shards embedding in parallel
)
def embed_shard(shard_uri: str) -> dict:
    """One GCS shard of chunks -> one GCS shard of vectors. Idempotent:
    a .done marker makes re-runs skip completed shards instantly."""
    import numpy as np
    from FlagEmbedding import BGEM3FlagModel
    from google.cloud import storage

    client = storage.Client()

    def blob_of(uri: str):
        bucket_name, path = uri.removeprefix("gs://").split("/", 1)
        return client.bucket(bucket_name).blob(path)

    out_uri = shard_uri.replace(".jsonl", ".vectors.jsonl")
    done_uri = out_uri + ".done"
    if blob_of(done_uri).exists():
        return {"shard": shard_uri, "status": "already_done"}

    rows = [json.loads(line) for line in
            blob_of(shard_uri).download_as_text().splitlines() if line.strip()]

    model = BGEM3FlagModel(MODEL, use_fp16=True)      # fp16 on H100: 2x throughput

    # sort by token_count so each batch holds similar lengths -> minimal padding
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("token_count", 512))
    results: dict[str, list[float]] = {}
    batch: list[int] = []
    batch_tokens = 0

    def flush() -> None:
        nonlocal batch, batch_tokens
        if not batch:
            return
        texts = [rows[i]["embed_text"] for i in batch]
        dense = model.encode(texts, batch_size=len(texts),
                             max_length=512)["dense_vecs"]
        dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
        for i, vec in zip(batch, dense):
            results[rows[i]["chunk_id"]] = [round(float(x), 7) for x in vec]
        batch, batch_tokens = [], 0

    for i in order:
        n = rows[i].get("token_count", 512)
        if batch and batch_tokens + n > BATCH_MAX_TOKENS:
            flush()
        batch.append(i)
        batch_tokens += n
    flush()

    payload = "\n".join(
        json.dumps({"chunk_id": cid, "dense": vec, "model": MODEL})
        for cid, vec in results.items())
    blob_of(out_uri).upload_from_string(payload)
    blob_of(done_uri).upload_from_string("")           # marker LAST: crash-safe
    return {"shard": shard_uri, "status": "done", "chunks": len(results)}


@app.local_entrypoint()
def main(shard_prefix: str):
    """Fan every shard under the prefix across H100 containers."""
    listing = subprocess.run(
        ["gsutil", "ls", f"{shard_prefix}/*.jsonl"],
        capture_output=True, text=True, check=True).stdout.split()
    shards = [u for u in listing if not u.endswith(".vectors.jsonl")]
    print(f"{len(shards)} shards -> H100 fleet")
    for result in embed_shard.map(shards):
        print(result)
