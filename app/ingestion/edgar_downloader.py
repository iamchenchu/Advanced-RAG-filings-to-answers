"""
edgar_downloader.py

One-file end-to-end test:
  fetch submissions -> find latest 10-K -> download document -> upload to GCS
"""

import os
import requests
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# --- config from environment ---
SEC_USER_AGENT = os.environ["SEC_USER_AGENT"]   # "Your Name your-email@example.com"
GCS_BUCKET = os.environ["GCS_BUCKET"]

HEADERS = {"User-Agent": SEC_USER_AGENT}


def get_latest_10k(cik: int) -> dict:
    """Fetch a company's submissions and return metadata for its most recent 10-K."""
    cik_padded = f"{cik:010d}"
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    # These arrays are parallel: index i describes one filing across all of them.
    for form, accession, date, doc in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["filingDate"],
        recent["primaryDocument"],
    ):
        if form == "10-K":
            return {
                "cik": cik,
                "accession": accession,        # e.g. 0000320193-23-000106
                "filing_date": date,
                "primary_document": doc,        # e.g. aapl-20230930.htm
            }

    raise ValueError(f"No 10-K found for CIK {cik}")


def download_filing(meta: dict) -> bytes:
    """Download the primary document for a filing and return its raw bytes."""
    accession_nodashes = meta["accession"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{meta['cik']}/{accession_nodashes}/{meta['primary_document']}"
    )

    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    if not resp.content:                       # cheap validation
        raise ValueError(f"Empty document at {url}")

    return resp.content


def upload_to_gcs(data: bytes, meta: dict) -> str:
    """Write the document bytes to GCS and return the gs:// URI."""
    accession_nodashes = meta["accession"].replace("-", "")
    blob_name = f"filings/{meta['cik']}/{accession_nodashes}/{meta['primary_document']}"

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="text/html")

    return f"gs://{GCS_BUCKET}/{blob_name}"


if __name__ == "__main__":
    APPLE_CIK = 320193

    meta = get_latest_10k(APPLE_CIK)
    print(f"Found 10-K: {meta['accession']} ({meta['filing_date']})")

    data = download_filing(meta)
    print(f"Downloaded {len(data):,} bytes")

    uri = upload_to_gcs(data, meta)
    print(f"Uploaded -> {uri}")