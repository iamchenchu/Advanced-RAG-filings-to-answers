"""
clients/inference.py

The ONLY cross-system dependency: Distributed-Inference-Serving, reached over
plain OpenAI-compatible HTTP, never imported.

The degradation contract (PRD section 5) is enforced HERE, at the boundary:
every failure mode of the LLM becomes InferenceUnavailable, and the pipeline
maps that to answer=null + degraded="generation_unavailable" + citations
intact. Retrieval must keep working when generation is down — the Multi-Agent
consumer wants passages more often than prose anyway.

A small circuit breaker stops a dead inference cluster from adding
connect-timeout latency to every query: after N consecutive failures the
breaker opens and calls fail instantly for reset_s seconds.
"""

import time

import httpx

from app.config import get_settings


class InferenceUnavailable(Exception):
    pass


class _Breaker:
    def __init__(self, threshold: int = 5, reset_s: float = 30):
        self.threshold = threshold
        self.reset_s = reset_s
        self.failures = 0
        self.opened_at: float | None = None

    def check(self) -> None:
        if self.opened_at is not None:
            if time.monotonic() - self.opened_at < self.reset_s:
                raise InferenceUnavailable("circuit breaker open")
            self.opened_at = None               # half-open: allow one probe
            self.failures = self.threshold - 1

    def ok(self) -> None:
        self.failures = 0
        self.opened_at = None

    def fail(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


class InferenceClient:
    def __init__(self):
        s = get_settings()
        self.url = s.inference_url.rstrip("/")
        self.model = s.inference_model
        self.api_key = s.inference_api_key
        self.max_retries = s.inference_max_retries
        self.timeout = httpx.Timeout(
            connect=s.inference_connect_timeout_s,
            read=s.inference_first_token_timeout_s,   # non-stream: first byte of body
            write=10, pool=10,
        )
        self.total_timeout = s.inference_total_timeout_s
        self.breaker = _Breaker()

    def _headers(self) -> dict:
        h = {"X-Workload-Class": "rag"}              # routes to the throughput pool
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat(self, messages: list[dict], max_tokens: int = 1024) -> dict:
        """Non-streaming completion -> {"text": ..., "usage": {...}}.

        Retries happen ONLY before any output has been produced; a stream that
        already emitted tokens is never retried (it would duplicate the answer).
        Non-streaming calls are all-or-nothing, so retrying here is safe.
        """
        self.breaker.check()
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": max_tokens, "stream": False}
        last: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                resp = httpx.post(f"{self.url}/chat/completions", json=payload,
                                  headers=self._headers(), timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                self.breaker.ok()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage", {}),
                }
            except Exception as exc:                 # noqa: BLE001
                last = exc
        self.breaker.fail()
        raise InferenceUnavailable(str(last))


    def chat_stream(self, messages: list[dict], max_tokens: int = 1024):
        """Streaming completion: yields text deltas as they arrive.

        The retry discipline is the whole point of this method existing
        separately: a retry is permitted ONLY while zero tokens have been
        yielded. The moment the first delta goes out, any failure surfaces
        as-is - re-running a half-emitted answer would duplicate text the
        consumer already rendered.
        """
        import json as _json

        self.breaker.check()
        yielded_any = False
        attempts = 0
        while True:
            try:
                with httpx.stream(
                    "POST", f"{self.url}/chat/completions",
                    headers=self._headers(),
                    json={"model": self.model, "messages": messages,
                          "max_tokens": max_tokens, "stream": True},
                    timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload.strip() == "[DONE]":
                            self.breaker.ok()
                            return
                        delta = (_json.loads(payload)["choices"][0]
                                 .get("delta", {}).get("content"))
                        if delta:
                            yielded_any = True
                            yield delta
                    self.breaker.ok()
                    return
            except Exception as exc:              # noqa: BLE001
                if yielded_any:
                    # mid-stream failure: never retry, never duplicate
                    raise InferenceUnavailable(f"stream broke mid-answer: {exc}") from exc
                self.breaker.fail()
                attempts += 1
                if attempts > self.max_retries:
                    raise InferenceUnavailable(str(exc)) from exc
                self.breaker.check()


_client: InferenceClient | None = None


def get_inference() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient()
    return _client
