"""
modal_app/embed_h100.py — the H100 embedding fleet, Modal-volume edition.

Data path (no GCP credentials on Modal — shards travel via a Modal Volume):

    scripts/make_shards_local.py   Postgres -> data/shards/*.jsonl
    modal volume put               data/shards -> volume /in
    modal run modal_app/embed_h100.py        (this file: /in -> H100 -> /out)
    modal volume get               volume /out -> data/vectors
    scripts/load_vectors.py        data/vectors -> chunk_vectors + outbox

Fleet shape: up to 8 H100 containers; each loads BGE-M3 (fp16) ONCE via a
container-lifecycle hook, then chews through shards. Token-sorted dynamic
batches (~16k tokens per forward pass) keep padding waste minimal — the same
trick TEI uses. Every shard writes a .done marker LAST, so a re-run after any
failure skips finished work: the whole stage is resumable at shard level.

Output per shard: <name>.ids.json + <name>.f16.npy (row i of the matrix is
the L2-normalized 1024-dim vector for ids[i]).

Cost math at 2026 Modal pricing (~$3.95/H100-hour): ~1.1M chunks at
~4-6k chunks/min/GPU ≈ 3-5 GPU-hours ≈ $12-20 total, ~30-40 min wall clock
on 8 containers.
"""

import json

import modal

app = modal.App("rag-embed-h100")
volume = modal.Volume.from_name("rag-embed-shards", create_if_missing=True)

MODEL = "BAAI/bge-m3"
BATCH_MAX_TOKENS = 16384
VOL = "/vol"


def _download_model() -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL)


image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers/peft MUST be pinned: FlagEmbedding 1.3.5 imports symbols
    # (is_torch_fx_available) that newer transformers removed — unpinned, the
    # image builds fine and then dies at import time on the GPU.
    .pip_install("FlagEmbedding==1.3.5", "transformers==4.45.2", "peft==0.13.2",
                 "torch", "numpy", "huggingface_hub")
    .run_function(_download_model)      # bake weights into the image: containers
)                                       # start ready instead of downloading 2.3GB


@app.cls(
    image=image,
    gpu="H100",
    timeout=3600,
    retries=2,
    max_containers=8,
    volumes={VOL: volume},
)
class Embedder:
    @modal.enter()
    def load(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(MODEL, use_fp16=True)

    @modal.method()
    def embed_shard(self, shard_name: str) -> dict:
        import numpy as np

        in_path = f"{VOL}/in/{shard_name}"
        out_ids = f"{VOL}/out/{shard_name.removesuffix('.jsonl')}.ids.json"
        out_npy = f"{VOL}/out/{shard_name.removesuffix('.jsonl')}.f16.npy"
        done = f"{VOL}/out/{shard_name.removesuffix('.jsonl')}.done"

        import os
        volume.reload()                       # see the freshest volume state
        if os.path.exists(done):
            return {"shard": shard_name, "status": "already_done"}
        os.makedirs(f"{VOL}/out", exist_ok=True)

        rows = [json.loads(line) for line in open(in_path) if line.strip()]

        # token-sorted batching: similar lengths per forward pass -> ~no padding
        order = sorted(range(len(rows)), key=lambda i: rows[i].get("token_count", 512))
        vectors: dict[str, "np.ndarray"] = {}
        batch: list[int] = []
        batch_tokens = 0

        def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            texts = [rows[i]["embed_text"] for i in batch]
            dense = self.model.encode(texts, batch_size=len(texts),
                                      max_length=512)["dense_vecs"]
            dense = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
            for i, vec in zip(batch, dense):
                vectors[rows[i]["chunk_id"]] = vec.astype(np.float16)
            batch, batch_tokens = [], 0

        for i in order:
            n = rows[i].get("token_count", 512)
            if batch and batch_tokens + n > BATCH_MAX_TOKENS:
                flush()
            batch.append(i)
            batch_tokens += n
        flush()

        ids = list(vectors.keys())
        matrix = np.stack([vectors[c] for c in ids]) if ids else np.zeros((0, 1024), "float16")
        with open(out_npy, "wb") as fh:
            np.save(fh, matrix)
        with open(out_ids, "w") as fh:
            json.dump(ids, fh)
        open(done, "w").close()               # marker LAST: crash-safe resume
        volume.commit()
        return {"shard": shard_name, "status": "done", "chunks": len(ids)}


@app.local_entrypoint()
def main():
    """Embed every shard under /in that has no .done marker yet."""
    import os
    # list the volume from the local side
    shards = sorted(
        e.path.removeprefix("in/") for e in volume.listdir("in")
        if e.path.endswith(".jsonl"))
    done = {e.path.removeprefix("out/").removesuffix(".done")
            for e in _safe_listdir("out") if e.path.endswith(".done")}
    todo = [sh for sh in shards if sh.removesuffix(".jsonl") not in done]
    print(f"{len(shards)} shards on volume, {len(shards) - len(todo)} done, "
          f"{len(todo)} to embed")
    total = 0
    embedder = Embedder()
    for result in embedder.embed_shard.map(todo):
        total += result.get("chunks", 0)
        print(result)
    print(f"embedded {total:,} chunks")


def _safe_listdir(path: str):
    try:
        return volume.listdir(path)
    except Exception:                          # noqa: BLE001 — /out not yet created
        return []
