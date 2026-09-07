#!/bin/bash
# overnight_backfill.sh — runs the entire remaining pipeline unattended:
#   wait for downloads -> wait for parse drain -> shard export -> Modal volume
#   -> full H100 fleet -> vectors back -> Postgres -> Qdrant -> golden eval
# Every stage is idempotent/resumable; re-running this script skips done work.
set -x
cd /Users/mekalathuruchenchaiah/Desktop/Big-Project/Advanced-RAG-System
PY=/Users/mekalathuruchenchaiah/Desktop/Big-Project/.venv/bin/python
MODAL=/Users/mekalathuruchenchaiah/Desktop/Big-Project/.venv/bin/modal
PGURL="postgresql://rag:changeme@localhost:5432/rag_db"

echo "=== [1/9] waiting for all download batches ==="
until grep -q "ALL DOWNLOAD BATCHES COMPLETE" logs/download.log; do sleep 60; done
echo "downloads complete: $(grep -c '^->' logs/download.log) new filings this run"

echo "=== [2/9] waiting for parse queue to drain ==="
while true; do
  n=$(psql "$PGURL" -tA -c "SELECT count(*) FROM ingestion_jobs WHERE status IN ('pending','processing')")
  [ "$n" = "0" ] && break
  sleep 60
done
psql "$PGURL" -tA -c "SELECT 'final corpus: '||count(*)||' chunks from '||(SELECT count(*) FROM documents WHERE status='ready')||' documents' FROM document_chunks"

echo "=== [3/9] exporting shards ==="
$PY scripts/make_shards_local.py

echo "=== [4/9] uploading shards to Modal volume ==="
$MODAL volume put --force rag-embed-shards data/shards /in

echo "=== [5/9] H100 fleet run ==="
$MODAL run modal_app/embed_h100.py

echo "=== [6/9] downloading vectors ==="
rm -rf data/vectors && mkdir -p data/vectors
$MODAL volume get --force rag-embed-shards /out data/vectors
mv data/vectors/out/* data/vectors/ 2>/dev/null; rmdir data/vectors/out 2>/dev/null

echo "=== [7/9] loading vectors into Postgres ==="
$PY scripts/load_vectors.py

echo "=== [8/9] syncing to Qdrant (both collections) ==="
$PY workers/outbox_worker.py --once

echo "=== [9/9] evaluations ==="
$PY evaluation/build_golden.py
$PY evaluation/run_golden.py
$PY evaluation/run_eval.py --config bge-m3-full
echo "=== OVERNIGHT BACKFILL COMPLETE ==="
