# Phase A (Kubernetes) - results

The RAG system, designed for Kubernetes from day one, now runs on it. Verified
live on a local `kind` (Kubernetes IN Docker) cluster, 2026-09-07.

## What runs

| Workload | Kind | Replicas | Notes |
|---|---|---|---|
| rag-api | Deployment + Service | 2 | serves retrieval + streaming in-cluster |
| ingestion-worker | Deployment | 1-6 (KEDA) | stateless SKIP LOCKED pollers |
| reaper | CronJob | */10 min | `concurrencyPolicy: Forbid` |
| Postgres / Qdrant / Ollama | on host | - | reached via host.docker.internal |

Stateful stores stay on the host for the local demo; they become StatefulSets
(or managed services) on real GKE (Google Kubernetes Engine).

## The proofs

**1. Probe design does real work.** `/health` (liveness) always returns 200;
`/ready` (readiness) returns 503 only when Postgres is unreachable. A pod with
a degraded Qdrant or LLM stays IN the Service and serves degraded answers -
exactly the intent. The `startupProbe` (200s budget) shields the slow
ONNX-load + migrations boot; before it was tuned, liveness restarted pods
mid-boot (3 restarts) - a textbook probe-ordering bug, caught and fixed.

**2. Idempotency held through chaos.** `document_chunks` stayed at exactly
**1,736,942** through: two rolling deploys, a `--force --grace-period=0` kill
of a worker mid-job, and a 60-job autoscaling storm. The killed worker's jobs
were reassigned by `FOR UPDATE SKIP LOCKED`; the Deployment replaced the pod.

**3. Queue-depth autoscaling works.** KEDA reads `rag_ingestion_jobs_pending`
from Prometheus. A 60-job backlog scaled workers **2 -> 4 -> 6** within ~45s,
then drained. This is the right signal (CPU% would miss a queue backing up on
an idle-CPU worker) - the lesson from interview-prep/04 made real.

## Known limitation (honest)

In-cluster ingestion workers cannot fetch `gs://` filings with the user's
Application Default Credentials - the ADC file does not mount as a usable
service identity. Re-parsing therefore fails in `kind` (jobs dead-letter
cleanly, losing no data). The fix on GKE is Workload Identity (a k8s service
account bound to a GCP service account, keyless); locally, mount a real
service-account key. Retrieval needs no GCS access, so the API tier is
unaffected and fully functional in-cluster.

## Run it

```bash
kind create cluster --config deploy/k8s/kind-config.yaml
docker build -t rag-api:local . && kind load docker-image rag-api:local --name rag
bash deploy/k8s/apply.sh          # namespace, config, secrets (imperative), workloads
kubectl -n llm-rag get pods
```

## Next on this track

GKE with Workload Identity (closes the GCS limitation + starts the multi-cloud
story), a real Ingress instead of port-forward, and StatefulSets for
Postgres/Qdrant.
