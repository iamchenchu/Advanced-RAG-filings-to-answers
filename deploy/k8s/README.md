# Kubernetes deployment (Phase A) — local kind cluster

Brings the RAG system up on Kubernetes. Proves the production shape: stateless
API + workers as pods, databases stay outside the cluster, config via
ConfigMap/Secret, model-aware probes, and autoscaling on the RIGHT signals.

## Layout

| File | What |
|---|---|
| `kind-config.yaml` | 1-node cluster; mounts the host HF model cache so API pods don't re-download 2.3 GB |
| `00-namespace.yaml` | namespace `llm-rag` |
| `01-configmap.yaml` | non-secret config; points pods at host Postgres/Qdrant/Ollama/Prometheus |
| `02-secret.template.yaml` | secret TEMPLATE; `apply.sh` substitutes `DATABASE_URL` at apply time (real secret never committed) |
| `10-rag-api.yaml` | API Deployment (2 replicas) + Service; liveness=/health, readiness=/ready, startup probe for ONNX load |
| `20-ingestion-worker.yaml` | worker Deployment (defer backend; SKIP LOCKED means replicas share work) |
| `30-reaper-cronjob.yaml` | reaper every 10 min, `concurrencyPolicy: Forbid` |
| `40-hpa-api.yaml` | CPU HPA for the API (needs metrics-server) |
| `50-keda-workers.yaml` | KEDA ScaledObject: workers scale on `rag_ingestion_jobs_pending` from Prometheus (needs KEDA) |

## Run

```bash
kind create cluster --config kind-config.yaml
docker build -t rag-api:local ..              # from the project root
kind load docker-image rag-api:local --name rag
./apply.sh
kubectl -n llm-rag get pods -w
kubectl -n llm-rag port-forward svc/rag-api 8080:80   # then curl localhost:8080
```

## Two real incidents hit while building this (the teaching value)

1. **CrashLoopBackOff — `TimeoutError` on startup.** Pods couldn't reach the
   host's Postgres. `host.docker.internal` doesn't resolve inside kind pods
   (only in Docker Desktop's own containers). Fix: a `hostAliases` entry
   mapping `host.docker.internal` to the Docker host-gateway IPv4
   (`192.168.65.254`). First attempt used the IPv6 address and still crashed —
   the readiness probe (fail-fast on Postgres) is exactly what surfaced it, and
   the startup probe stopped a slow ONNX load from being mistaken for a crash.

2. **Worker pods dead-letter every job — `DefaultCredentialsError`.** In-cluster
   workers parse `gs://` filings, which needs Google Cloud credentials the pod
   doesn't have. On the laptop, host processes inherit Application Default
   Credentials; a pod does not. This is the real Workload Identity lesson
   (doc 07): cloud access in-cluster must be granted explicitly — via a mounted
   service-account key (dev) or Workload Identity (GKE). Until then, in-cluster
   workers are scaled to 0 and ingestion runs on the host; the API pods serve
   queries fine (they only touch Postgres + Qdrant, which are reachable).

## Verified working

- 2 API pods Running; a query through the **Service** returns cited passages.
- Rolling update observed (old ReplicaSet drained as new one came up).
- **Failure drill:** killed a worker pod mid-ingest with `--grace-period=0
  --force`; data integrity held exactly — 6,947 documents, 1,736,942 chunks,
  no loss, no duplication (the SKIP-LOCKED + reconcile + deterministic-id
  design doing its job under a real kill).
