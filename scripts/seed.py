"""
scripts/seed.py — register the Apple FY2025 10-K (already in GCS) through the
real API path, exactly the way any client would:

    POST /api/v1/documents  ->  202 {document_id, job_id}

    python scripts/seed.py                     # API on localhost:8000
    python scripts/seed.py --api http://host:8000
"""

import argparse

import httpx

APPLE = {
    "name": "Apple Inc. 10-K FY2025",
    "gcs_uri": "gs://llm-platform-services-bucket/filings/320193/000032019325000079/aapl-20250927.htm",
    "source_type": "html",
    "tags": ["10-K", "tech"],
    "meta": {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "cik": "320193",
        "fiscal_year": "2025",
        "filing_date": "2025-10-31",
        "form_type": "10-K",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    resp = httpx.post(f"{args.api}/api/v1/documents", json=APPLE, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    print(f"accepted: document_id={body['document_id']} job_id={body['job_id']}")
    print("now run:  python workers/ingestion_worker.py --once")
    print("then:     python workers/outbox_worker.py --once")


if __name__ == "__main__":
    main()
