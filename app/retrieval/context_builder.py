"""
retrieval/context_builder.py

Reranked passages -> the exact context block the LLM sees + citation records
the client sees. Responsibilities (PRD 3.8), in order:

  1. near-duplicate suppression  — cosine >= DEDUPE_THRESHOLD never enters
     the context twice (overlapping windows make this common)
  2. source diversity            — at most MAX_CHUNKS_PER_DOCUMENT chunks per
     document, so one filing cannot monopolize the context
  3. token budget                — stop before MAX_CONTEXT_TOKENS minus the
     answer reserve; oversized is PREVENTED, not truncated after the fact
  4. citation ids                — [1..n] assigned here and only here; every
     id maps to a real chunk BY CONSTRUCTION, which is how the "100% of
     citations resolve" acceptance criterion is met
"""

import numpy as np

from app.config import get_settings


def build_context(candidates: list[dict]) -> dict:
    """candidates: reranked (or fused) chunks, best first, each with
    text / token_count / document metadata, optionally vector + scores.

    -> {"context_text", "citations", "context_tokens", "n_documents", "dropped"}
    """
    s = get_settings()
    budget = s.max_context_tokens - s.context_reserve_tokens

    selected: list[dict] = []
    vectors: list[np.ndarray] = []
    per_doc: dict[str, int] = {}
    tokens_used = 0
    dropped = {"duplicate": 0, "doc_cap": 0, "budget": 0}

    for cand in candidates:
        n_tokens = int(cand.get("token_count") or max(1, len(cand["text"]) // 4))

        if tokens_used + n_tokens > budget:
            dropped["budget"] += 1
            continue          # a smaller later chunk may still fit

        doc = str(cand.get("document_id"))
        if per_doc.get(doc, 0) >= s.max_chunks_per_document:
            dropped["doc_cap"] += 1
            continue

        vec = cand.get("vector")
        if vec is not None and vectors:
            v = np.asarray(vec, dtype=np.float32)
            v = v / (np.linalg.norm(v) or 1.0)
            if max(float(v @ w) for w in vectors) >= s.dedupe_threshold:
                dropped["duplicate"] += 1
                continue

        if vec is not None:
            v = np.asarray(vec, dtype=np.float32)
            vectors.append(v / (np.linalg.norm(v) or 1.0))
        selected.append(cand)
        per_doc[doc] = per_doc.get(doc, 0) + 1
        tokens_used += n_tokens

    blocks: list[str] = []
    citations: list[dict] = []
    for i, cand in enumerate(selected, start=1):
        label = " | ".join(str(part) for part in [
            cand.get("document_name"),
            f"Item {cand['section']}. {cand.get('section_title') or ''}".strip()
            if cand.get("section") else None,
        ] if part)
        blocks.append(f"[{i}] ({label})\n{cand['text']}")
        citations.append({
            "id": i,
            "document_id": str(cand.get("document_id")),
            "document_name": cand.get("document_name"),
            "page_number": cand.get("page_number"),
            "section": cand.get("section"),
            "chunk_id": str(cand["chunk_id"]),
            "score": round(float(cand.get("rerank_score",
                            cand.get("fused_score", cand.get("score", 0.0)))), 4),
            "text": cand["text"],
        })

    return {
        "context_text": "\n\n".join(blocks),
        "citations": citations,
        "context_tokens": tokens_used,
        "n_documents": len(per_doc),
        "dropped": dropped,
    }
