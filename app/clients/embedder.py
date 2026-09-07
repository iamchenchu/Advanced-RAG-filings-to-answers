"""
clients/embedder.py

Dense embeddings behind one interface, three backends:

  tei        HTTP to a text-embeddings-inference server (production; also the
             shape the Modal H100 backfill mimics)
  fastembed  in-process ONNX on CPU — real semantics, no service, no torch.
             This is the local-dev backend.
  hash       deterministic garbage — unit tests that need vectors but not
             meaning, and offline CI.
  defer      no vectors at all — ingestion inserts chunks with dense=NULL and
             the Modal H100 backfill fills them in later. Parse-now/embed-later
             is what lets 5,000 filings parse locally while the GPU fleet does
             the only part that actually needs a GPU.
  onnx       BAAI's official ONNX export of bge-m3, run in-process through
             onnxruntime (native on Apple Silicon). This is the local
             QUERY-time backend once the corpus is embedded with bge-m3:
             queries and documents must share the model, fastembed has no
             bge-m3, and TEI ships no arm64 CPU image — this path does both.

The invariant that matters: QUERIES AND DOCUMENTS MUST BE EMBEDDED BY THE
SAME MODEL. Vectors from different models live in unrelated spaces; nearest
neighbors across them are noise. That is why embedding_model is recorded on
every chunk row — stale vectors are findable when the model changes.
"""

import hashlib

import httpx
import numpy as np

from app.config import get_settings


class EmbedderUnavailable(Exception):
    """Raised when no dense vector can be produced. The pipeline catches this
    and degrades to sparse-only ("dense_unavailable"), never a 500."""


class _TEIBackend:
    def __init__(self, url: str, timeout: float, retries: int):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def embed(self, texts: list[str]) -> np.ndarray:
        last: Exception | None = None
        for _ in range(self.retries):
            try:
                resp = httpx.post(f"{self.url}/embed", json={"inputs": texts},
                                  timeout=self.timeout)
                resp.raise_for_status()
                return np.asarray(resp.json(), dtype=np.float32)
            except Exception as exc:          # noqa: BLE001 — every failure = retry
                last = exc
        raise EmbedderUnavailable(f"TEI at {self.url}: {last}")


class _FastembedBackend:
    _model = None                             # class-level: load ONNX once per process

    def __init__(self, model_name: str):
        # every failure (download, ONNX init, inference) must surface as
        # EmbedderUnavailable — the pipeline's degradation contract catches
        # exactly that type; a raw ONNX error would 500 the query instead
        try:
            if _FastembedBackend._model is None:
                from fastembed import TextEmbedding
                _FastembedBackend._model = TextEmbedding(model_name)
        except Exception as exc:              # noqa: BLE001
            raise EmbedderUnavailable(f"fastembed load: {exc}") from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            return np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        except Exception as exc:              # noqa: BLE001
            raise EmbedderUnavailable(f"fastembed embed: {exc}") from exc


class _OnnxBackend:
    """BGE-M3 dense via BAAI's official onnx/ export. Dense == the CLS token
    of the last hidden state (normalization happens in Embedder.embed)."""

    _session = None
    _tok = None
    _pad_id = 1                    # XLM-RoBERTa <pad>

    def __init__(self, model_name: str):
        try:
            if _OnnxBackend._session is None:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer as HFTokenizer

                model_path = hf_hub_download(model_name, "onnx/model.onnx")
                hf_hub_download(model_name, "onnx/model.onnx_data")
                _OnnxBackend._session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"])
                tok = HFTokenizer.from_pretrained(model_name)
                tok.enable_truncation(max_length=512)
                tok.no_padding()
                _OnnxBackend._tok = tok
                pad = tok.token_to_id("<pad>")
                if pad is not None:
                    _OnnxBackend._pad_id = pad
        except Exception as exc:              # noqa: BLE001
            raise EmbedderUnavailable(f"onnx load: {exc}") from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            encs = self._tok.encode_batch(texts)     # with special tokens
            width = max(len(e.ids) for e in encs)
            ids = np.full((len(encs), width), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(encs), width), dtype=np.int64)
            for i, e in enumerate(encs):
                ids[i, : len(e.ids)] = e.ids
                mask[i, : len(e.ids)] = 1
            inputs = {"input_ids": ids, "attention_mask": mask}
            wanted = {i.name for i in self._session.get_inputs()}
            # BAAI's export names a pooled output explicitly — ask for it by
            # name instead of trusting output order, and fall back to CLS
            # pooling only if this export lacks it
            names = [o.name for o in self._session.get_outputs()]
            target = ["sentence_embedding"] if "sentence_embedding" in names else None
            out = self._session.run(
                target, {k: v for k, v in inputs.items() if k in wanted})[0]
            if out.ndim == 3:                        # token embeddings
                out = out[:, 0, :]                   # CLS pooling = bge-m3 dense
            return np.asarray(out, dtype=np.float32)
        except Exception as exc:              # noqa: BLE001
            raise EmbedderUnavailable(f"onnx embed: {exc}") from exc


class _DeferBackend:
    """Produces NO vectors. Ingestion stores dense=NULL; make_shards later
    finds exactly those rows for the GPU backfill."""

    def embed(self, texts: list[str]):
        return None


class _HashBackend:
    """Deterministic pseudo-vectors. Same text -> same vector, always."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


class Embedder:
    def __init__(self):
        s = get_settings()
        self.model_name = s.embedding_model
        self.dim = s.embedding_dim
        self.batch_size = s.embedding_batch_size
        if s.embedding_backend == "tei":
            self.backend = _TEIBackend(s.embedding_url, s.embedding_timeout_s,
                                       s.embedding_max_retries)
        elif s.embedding_backend == "fastembed":
            self.backend = _FastembedBackend(s.embedding_model)
        elif s.embedding_backend == "onnx":
            self.backend = _OnnxBackend(s.embedding_model)
        elif s.embedding_backend == "defer":
            self.backend = _DeferBackend()
        else:
            self.backend = _HashBackend(s.embedding_dim)

    def embed(self, texts: list[str]) -> np.ndarray | None:
        """Batched, L2-normalized. Cosine similarity == dot product after this,
        which is what both Qdrant (cosine distance) and MMR assume.
        Returns None under the defer backend (vectors arrive via the backfill)."""
        if isinstance(self.backend, _DeferBackend):
            return None
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            chunks.append(self.backend.embed(texts[i : i + self.batch_size]))
        vecs = np.vstack(chunks)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
