#!/bin/bash
# Applies the whole stack to the kind cluster in dependency order.
# Secrets are substituted here from the environment, never committed.
set -euo pipefail
cd "$(dirname "$0")"
DB_URL="${RAG_DATABASE_URL:-postgresql://rag:changeme@host.docker.internal:5432/rag_db}"

kubectl apply -f 00-namespace.yaml
sed "s|__DATABASE_URL__|$DB_URL|" 02-secret.template.yaml | kubectl apply -f -
kubectl apply -f 01-configmap.yaml
kubectl apply -f 10-rag-api.yaml -f 20-ingestion-worker.yaml -f 30-reaper-cronjob.yaml
kubectl apply -f 40-hpa-api.yaml
# 50 (KEDA) applied separately once KEDA is installed
echo "applied. watch: kubectl -n llm-rag get pods -w"
