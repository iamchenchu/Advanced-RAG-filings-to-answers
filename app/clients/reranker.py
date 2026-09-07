"""
clients/reranker.py

Cross-encoder reranking. A bi-encoder (the embedding model) scores query and
document INDEPENDENTLY and meets them in vector space — fast, but it can only
say "these are about the same topic." A cross-encoder reads query+passage
TOGETHER through one transformer and scores actual relevance. It is the
single largest quality jump in the pipeline (PRD 3.5) and too slow to run
over the whole corpus — which is why it only sees the fused top ~100.

Backends: tei (HTTP, production) | fastembed (local ONNX) | none.
Any failure -> RerankUnavailable -> the pipeline returns fusion order and
flags "rerank_skipped". Skipping quality beats serving a 500.
"""

import httpx

from app.config import get_settings


class RerankUnavailable(Exception):
    pass


class _TEIRerank:
    def __init__(self, url: str, timeout: float):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def scores(self, query: str, texts: list[str]) -> list[float]:
        try:
            resp = httpx.post(f"{self.url}/rerank",
                              json={"query": query, "texts": texts, "raw_scores": True},
                              timeout=self.timeout)
            resp.raise_for_status()
        except Exception as exc:              # noqa: BLE001
            raise RerankUnavailable(str(exc)) from exc
        ranked = resp.json()                  # [{index, score}]
        out = [0.0] * len(texts)
        for item in ranked:
            out[item["index"]] = float(item["score"])
        return out


class _FastembedRerank:
    _model = None

    def __init__(self, model_name: str):
        # same rule as the embedder: any failure = RerankUnavailable, so the
        # pipeline degrades to fusion order instead of 500ing the query
        try:
            if _FastembedRerank._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                _FastembedRerank._model = TextCrossEncoder(model_name)
        except Exception as exc:              # noqa: BLE001
            raise RerankUnavailable(f"fastembed load: {exc}") from exc

    def scores(self, query: str, texts: list[str]) -> list[float]:
        try:
            return [float(s) for s in self._model.rerank(query, texts)]
        except Exception as exc:              # noqa: BLE001
            raise RerankUnavailable(f"fastembed rerank: {exc}") from exc


class Reranker:
    def __init__(self):
        s = get_settings()
        self.enabled = s.reranker_enabled and s.reranker_backend != "none"
        self.top_n = s.reranker_top_n
        self.return_n = s.reranker_return_n
        if not self.enabled:
            self.backend = None
        elif s.reranker_backend == "tei":
            self.backend = _TEIRerank(s.reranker_url, s.reranker_timeout_s)
        else:
            self.backend = _FastembedRerank(s.reranker_model)

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """candidates: [{chunk_id, text, ...}] in fusion order.
        Returns top return_n by cross-encoder score, score attached —
        the recorded score is what makes rerank quality debuggable (PRD 3.5).
        """
        if not self.enabled or not candidates:
            raise RerankUnavailable("reranker disabled")
        pool = candidates[: self.top_n]
        scores = self.backend.scores(query, [c["text"] for c in pool])
        for c, score in zip(pool, scores):
            c["rerank_score"] = score
        pool.sort(key=lambda c: c["rerank_score"], reverse=True)
        return pool          # caller slices to its top_k; return_n is a default


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
