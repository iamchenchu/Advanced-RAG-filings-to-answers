# Advanced-RAG-System

Production-style RAG over SEC 10-K filings: hybrid retrieval (dense HNSW +
hand-written BM25), RRF fusion, MMR, cross-encoder reranking, metadata
pre-filtering, citation-aware generation — every stage independently measurable.

- **Start here:** [`explainer.md`](explainer.md) — file-by-file walkthrough, data
  flow, and exactly what to run.
- **Learn the concepts:** [`COMPONENTS.md`](COMPONENTS.md) — every component explained
  from scratch: what it is, why we chose it, alternatives, worked examples (RRF, MMR,
  reranking, HNSW, BM25, chunking, schema design).
- **Spec:** [`PRD.md`](PRD.md)
- **Quick start:** `python scripts/init_db.py`, then follow "What to run" in the explainer.
