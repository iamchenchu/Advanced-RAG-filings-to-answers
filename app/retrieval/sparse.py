"""
retrieval/sparse.py

Hand-written BM25 as Qdrant sparse vectors (PRD 3.4 — writing it is the point).

Where each half of the score lives:

  INDEX time (outbox worker):   tf_weight(term, chunk) =
      tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
    — term-frequency saturation (k1) and length normalization (b).

  QUERY time (here):            idf(term) =
      ln( (N - df + 0.5) / (df + 0.5) + 1 )
    — rarity. Computed from live corpus statistics in rag_db, so a term that
    becomes common as the corpus grows automatically loses weight.

Qdrant's sparse dot product multiplies the two halves back together:
  score = Σ_t  tf_weight(t, chunk) * idf(t)  ==  BM25.

Why sparse at all, next to a good dense model: exact terms. Tickers, "Item
1A", "$391,035 million", accession numbers, names. Dense embeddings blur
those into topic-space; BM25 matches them literally.

Caveat this file accepts: avgdl at index time is a snapshot. As the corpus
grows it drifts; requeue_all_chunks() re-syncs every point with fresh stats.
"""

import hashlib
import math
import re
from collections import Counter

# lowercase word tokens; keeps numbers ("10-K" -> "10", "k"), drops punctuation
_TOKEN = re.compile(r"[a-z0-9]+")

# tiny stopword list: only words so common their BM25 weight is pure noise.
# deliberately small — "risk" or "company" are stopwords in normal English but
# SIGNAL in a 10-K corpus. When in doubt, keep the word.
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with".split()
)


def analyze(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def term_id(term: str) -> int:
    """Stable 31-bit id for a term. Must be identical in every process forever
    (index writer and query side must agree), so: blake2b, not Python hash()
    which is salted per process."""
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big") & 0x7FFFFFFF


def encode_document(text: str) -> tuple[dict[int, int], list[str], int]:
    """-> ({term_id: tf}, [term strings, same order], doc_len).
    Raw counts only — weighting happens at sync time with live avgdl."""
    terms = analyze(text)
    counts = Counter(terms)
    tf: dict[int, int] = {}
    words: list[str] = []
    for term, n in counts.items():
        tid = term_id(term)
        if tid not in tf:               # rare 31-bit collision: first term wins
            tf[tid] = n
            words.append(term)
    return tf, words, len(terms)


def index_weights(tf: dict[int, int], doc_len: int, avgdl: float,
                  k1: float, b: float) -> dict[int, float]:
    """The tf half of BM25, applied when a chunk is synced to Qdrant."""
    if avgdl <= 0:
        avgdl = float(doc_len or 1)
    norm = k1 * (1 - b + b * doc_len / avgdl)
    return {tid: (n * (k1 + 1)) / (n + norm) for tid, n in tf.items()}


def query_weights(query: str, dfs: dict[int, int], n_chunks: int) -> dict[int, float]:
    """The idf half, applied to the query. Terms the corpus has never seen
    get no weight — they cannot match anything anyway."""
    weights: dict[int, float] = {}
    for term in set(analyze(query)):
        tid = term_id(term)
        df = dfs.get(tid, 0)
        if df <= 0 or n_chunks <= 0:
            continue
        idf = math.log((n_chunks - df + 0.5) / (df + 0.5) + 1.0)
        if idf > 0:
            weights[tid] = idf
    return weights
