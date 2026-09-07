# Case studies - the real stories, told for interviews

Four true stories from this system, each in the shape a behavioral or
system-design interview wants: situation, decision, numbers, lesson. Every
number is reproducible from evaluation/results/ and the git history.

---

## 1. The MVP-to-production migration that required zero re-architecture

**Situation.** Development started with a 384-dimension CPU embedding model
(bge-small) on 10 documents. Production needed BGE-M3 (1024 dimensions) across
6,947 filings - normally a rebuild-everything moment.

**Decision made early that paid off.** Two columns existed from day one:
`embedding_model` stamped on every vector row, and a chunking config-hash baked
into every chunk id. So "migrate models" became a query - "find rows whose
model differs" - not a rewrite.

**Numbers.** 1,736,942 chunks re-embedded on 8 rented H100 GPUs in under an
hour for ~$30 (vs ~20 hours of laptop CPU). Zero application code changed;
the same outbox worker synced the new vectors.

**Lesson.** Choose which decisions are reversible BEFORE the MVP: model choice
was made reversible by a version stamp; chunk-id determinism was made safe by
hashing the config into the id. Say this sentence in interviews: "we made the
expensive things incremental."

---

## 2. The config knob that existed but was never wired (the rescore bug)

**Situation.** INT8 scalar quantization was enabled to fit 1.73M vectors in
RAM. Latency improved 17x - but recall@10 flattened at exactly 0.940 no matter
how high ef_search went.

**Diagnosis.** A recall ceiling that ignores ef is the signature of
quantization error, which rescoring exists to fix. `.env` said
`QUANTIZATION_RESCORE=true` - but reading the search code showed the parameter
was never passed to the vector database. A knob that turns nothing.

**Fix and numbers.** One parameter block in `dense_search`
(rescore + 2x oversampling): recall@10 0.940 -> 0.980 at ef=32, latency
42ms -> 46ms. Full-precision quality at 16x less latency than the
pre-quantization baseline (726ms).

**Lesson.** Config that is not read is documentation, not behavior. The only
reason this was caught: every change gets measured against a ground-truth
collection. "We found it because we measure" beats "we were careful."

---

## 3. The quality stage that made quality worse (the reranker finding)

**Situation.** The pipeline included a cross-encoder reranker - the textbook
"quality" stage. On the full corpus, the labeled golden evaluation showed the
reranked results were WORSE than raw dense retrieval: hit@5 0.600 vs 0.667
(and at the 18-query scale, 0.444 vs 0.611).

**Diagnosis.** The local reranker (ms-marco-MiniLM) is a 90MB model from 2021
trained on English web search - judging 2024-era financial text ranked by a
far stronger 1024-dim bi-encoder. The junior model was overruling the senior
one.

**Decision.** Keep the stage (the architecture is right), flag the model:
production spec pins the matched bge-reranker-v2-m3. Every stage records its
scores, which is the only reason a "quality" stage hurting was visible at all.

**Lesson.** Never assume a pipeline stage adds value because the pattern says
it should - stage-level measurement or it did not happen.

---

## 4. Kill -9 in production: the pod-kill drill and two real k8s incidents

**Situation.** Moving the system onto Kubernetes (kind cluster), three things
happened that read like an incident report - because they were.

**Incident A - CrashLoopBackOff on startup.** Pods timed out reaching
Postgres: `host.docker.internal` does not resolve inside kind pods. Fixed with
a hostAliases entry to the host-gateway IPv4 (the IPv6 attempt also failed -
worth remembering). The fail-fast readiness probe surfaced it in seconds.

**Incident B - every in-cluster ingestion job dead-lettered.** Worker pods
raised DefaultCredentialsError loading filings from GCS: host processes
inherit Google credentials, pods do not. That is the Workload Identity lesson
live: in-cluster cloud access must be granted explicitly. Contained by design -
3 attempts, dead-letter, document marked failed, nothing corrupted.

**The drill.** With real jobs mid-flight, a worker pod was force-killed
(--grace-period=0). Result: 6,947 documents and 1,736,942 chunks - exactly
the baseline, no loss, no duplication. SKIP LOCKED claiming + one-transaction
reconcile + deterministic ids means a killed worker costs nothing.

**Lesson.** Crash-safety is designed, then PROVEN by killing things on
purpose. An idempotent pipeline turns "pod died" from an incident into a
non-event - and the interview line is: "we force-killed workers mid-write and
counted rows afterward."
