"""
scripts/download_corpus.py — bulk EDGAR → GCS → registration, the corpus feeder.

    python scripts/download_corpus.py --tickers AAPL MSFT NVDA --years 2
    python scripts/download_corpus.py --tickers-file sp500.txt --years 5

Resumable by construction, like everything else in this pipeline:
  - a filing already in GCS is not re-downloaded (blob existence check)
  - registration dedupes on gcs_uri, so re-runs re-queue jobs harmlessly
    (and the ingestion worker no-ops on already-chunked documents)

SEC fair-use: identify yourself via SEC_USER_AGENT and stay well under
10 req/s — this script sleeps between EVERY EDGAR request. 5,000 filings
means roughly 10,001 requests (1 ticker map + N submissions + N documents);
at ~2 req/s that is under two hours, run it overnight or in tmux.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

SLEEP_S = 0.15                        # ~6 req/s, still under SEC's 10 req/s cap
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def sec_get(client: httpx.Client, url: str) -> httpx.Response:
    time.sleep(SLEEP_S)
    resp = client.get(url)
    resp.raise_for_status()
    return resp


def load_ticker_map(client: httpx.Client) -> dict[str, dict]:
    """SEC's official ticker -> {cik, title} map (one request for all of them)."""
    data = sec_get(client, TICKER_MAP_URL).json()
    return {row["ticker"].upper(): {"cik": int(row["cik_str"]), "company": row["title"]}
            for row in data.values()}


def list_10ks(client: httpx.Client, cik: int, max_filings: int) -> list[dict]:
    """The most recent `max_filings` 10-K filings for one company."""
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    recent = sec_get(client, url).json()["filings"]["recent"]
    out = []
    for form, accession, fdate, rdate, doc in zip(
            recent["form"], recent["accessionNumber"], recent["filingDate"],
            recent["reportDate"], recent["primaryDocument"]):
        if form != "10-K":
            continue
        out.append({"accession": accession, "filing_date": fdate,
                    "report_date": rdate, "primary_document": doc})
        if len(out) >= max_filings:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=[])
    ap.add_argument("--tickers-file", help="one ticker per line, # comments ok")
    ap.add_argument("--years", type=int, default=1, help="10-Ks per company (most recent first)")
    ap.add_argument("--api", default="http://localhost:8000",
                    help="RAG API for registration; empty string = download+upload only")
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]
    if args.tickers_file:
        tickers += [line.split("#")[0].strip().upper()
                    for line in Path(args.tickers_file).read_text().splitlines()
                    if line.split("#")[0].strip()]
    if not tickers:
        ap.error("no tickers given")

    s = get_settings()
    from google.cloud import storage
    bucket = storage.Client().bucket(s.gcs_bucket)

    client = httpx.Client(headers={"User-Agent": s.sec_user_agent}, timeout=60,
                          follow_redirects=True)
    ticker_map = load_ticker_map(client)

    done = skipped = failed = 0
    for ticker in tickers:
        info = ticker_map.get(ticker)
        if info is None:
            print(f"!! {ticker}: not in SEC ticker map, skipping")
            failed += 1
            continue
        cik, company = info["cik"], info["company"]
        try:
            filings = list_10ks(client, cik, args.years)
        except Exception as exc:                       # noqa: BLE001
            print(f"!! {ticker}: submissions fetch failed: {exc}")
            failed += 1
            continue

        for f in filings:
            acc = f["accession"].replace("-", "")
            blob_name = f"{s.gcs_prefix}{cik}/{acc}/{f['primary_document']}"
            gcs_uri = f"gs://{s.gcs_bucket}/{blob_name}"
            fiscal_year = (f["report_date"] or f["filing_date"])[:4]

            blob = bucket.blob(blob_name)
            if blob.exists():
                print(f"== {ticker} FY{fiscal_year}: already in GCS")
                skipped += 1
            else:
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
                       f"{f['primary_document']}")
                try:
                    data = sec_get(client, url).content
                    if not data:
                        raise ValueError("empty document")
                except Exception as exc:               # noqa: BLE001
                    print(f"!! {ticker} FY{fiscal_year}: download failed: {exc}")
                    failed += 1
                    continue
                blob.upload_from_string(data, content_type="text/html")
                print(f"-> {ticker} FY{fiscal_year}: {len(data):,} bytes -> {gcs_uri}")
                done += 1

            if args.api:
                # registration must NEVER kill a multi-hour download: the GCS
                # copy is the durable artifact, and re-running this script
                # re-registers anything missed (create_document dedupes).
                try:
                    resp = httpx.post(f"{args.api}/api/v1/documents", timeout=30, json={
                        "name": f"{company} 10-K FY{fiscal_year}",
                        "gcs_uri": gcs_uri,
                        "source_type": "html",
                        "tags": ["10-K"],
                        "meta": {"ticker": ticker, "company": company, "cik": str(cik),
                                 "fiscal_year": fiscal_year, "filing_date": f["filing_date"],
                                 "form_type": "10-K"},
                    })
                    resp.raise_for_status()
                except Exception as exc:               # noqa: BLE001
                    print(f"!! {ticker} FY{fiscal_year}: registration failed "
                          f"({type(exc).__name__}) — re-run this script to register")

    print(f"\ndownloaded {done}, already-had {skipped}, failed {failed}")
    if args.api:
        print("jobs queued — run: python workers/ingestion_worker.py --once")


if __name__ == "__main__":
    main()
