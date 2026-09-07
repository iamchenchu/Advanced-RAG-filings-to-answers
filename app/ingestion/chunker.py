"""
chunker.py

Turn a parsed 10-K into embedding-sized chunks that carry their own citation.

    gs:// filing
         |
    parser.html_to_text  ->  parser.split_into_items
         |
    strip page furniture   (running footers, PART banners, "Table of Contents")
         |
    sentence-aware token windows   (512 tokens, 64 overlap, real BGE-M3 tokenizer)
         |
    chunk records   (stable chunk_id, sha256, item, source_uri, token_count)
         |
    JSONL on disk   ->  next stages: embed -> Qdrant vectors + Postgres rows

Two rules this file exists to enforce:

  1. Token counts are MEASURED with the tokenizer that will embed the text,
     never estimated. A "512-token chunk" that is really 700 tokens gets
     silently truncated by the embedding server, and the tail of the chunk is
     retrievable in Postgres but invisible to search. That bug is undetectable
     without measuring here.

  2. chunk_id is DETERMINISTIC. Re-running this file on the same filing with
     the same config produces the same ids, so the Qdrant upsert is idempotent
     and a half-finished backfill can simply be re-run. The chunking config is
     hashed into the id, so changing chunk size produces a new id space instead
     of silently overwriting vectors built under the old settings.

Run:
    python app/ingestion/chunker.py                              # the Apple filing in GCS
    python app/ingestion/chunker.py --uri ./aapl-20250927.htm --ticker AAPL
    python app/ingestion/chunker.py --items 1A,7 --sample 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

# parser.py lives next to this file. Support both `python app/ingestion/chunker.py`
# and `python -m app.ingestion.chunker` from the project root.
try:
    from app.ingestion.parser import html_to_text, split_into_items, load_html
except ImportError:  # running the file directly
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parser import html_to_text, split_into_items, load_html  # type: ignore

load_dotenv()


# ---------------------------------------------------------------------------
# Config. Everything comes from .env so a chunking experiment is a config
# change, not a code change (PRD 3.2: chunk size is an experiment parameter).
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    return int(raw.split("#")[0].strip())        # tolerate "512   # comment"


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.split("#")[0].strip() or default


CHUNK_STRATEGY      = _env_str("CHUNK_STRATEGY", "section_aware")
CHUNK_SIZE_TOKENS   = _env_int("CHUNK_SIZE_TOKENS", 512)
CHUNK_OVERLAP_TOKENS= _env_int("CHUNK_OVERLAP_TOKENS", 64)
MIN_CHUNK_TOKENS    = _env_int("MIN_CHUNK_TOKENS", 32)
EMBEDDING_MODEL     = _env_str("EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_TENANT      = _env_str("DEFAULT_TENANT", "default")

CHUNKER_VERSION = "1"        # bump when the algorithm changes, not the config
SPECIAL_TOKENS = 2           # [CLS] and [SEP], added by the embedding server
SAFETY_TOKENS  = 4           # slack for tokens that merge across the header join


# ---------------------------------------------------------------------------
# Tokenizer
#
# Loads the actual BGE-M3 tokenizer (XLM-RoBERTa, 250k vocab). First call
# downloads ~17 MB to ~/.cache/huggingface and every later call is local.
# No torch, no transformers: the `tokenizers` Rust library is enough, which
# also makes this importable inside a lightweight ingestion worker image.
# ---------------------------------------------------------------------------

class Tokenizer:
    """Token counting with the same tokenizer the embedding server uses."""

    def __init__(self, model: str = EMBEDDING_MODEL):
        from tokenizers import Tokenizer as HFTokenizer

        self.model = model
        self._tok = HFTokenizer.from_pretrained(model)
        # Never let the tokenizer truncate or pad; we do the windowing ourselves.
        self._tok.no_truncation()
        self._tok.no_padding()

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_batch(self, texts: list[str]) -> list[int]:
        """Batch encode. ~10x faster than a Python loop over 1.5M sentences."""
        if not texts:
            return []
        encs = self._tok.encode_batch(texts, add_special_tokens=False)
        return [len(e.ids) for e in encs]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """(start, end) char span of every token. Used to cut oversized text
        on a real token boundary instead of mid-word."""
        return self._tok.encode(text, add_special_tokens=False).offsets


# ---------------------------------------------------------------------------
# Boilerplate removal
#
# Every page break in a 10-K leaves furniture in the flattened text: a running
# footer, a "Table of Contents" back-link, a PART banner. Left in place it gets
# embedded into hundreds of chunks, and because it is identical everywhere it
# pulls unrelated chunks toward each other in vector space.
# ---------------------------------------------------------------------------

BOILERPLATE = [
    # "Apple Inc. | 2025 Form 10-K | 18"  -- the running footer, once per page
    re.compile(r"[^|\n]{0,60}\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d{1,4}", re.I),
    # "Page 12 of 118"
    re.compile(r"\bPage\s+\d{1,4}\s+of\s+\d{1,4}\b", re.I),
    # the back-link dropped at every page break
    re.compile(r"\bTable\s+of\s+Contents\b", re.I),
    # a PART banner sitting immediately in front of an Item header
    re.compile(r"\bPART\s+(?:I{1,3}|IV)\b(?=\s+Item\b)", re.I),
    # a PART banner alone on its line
    re.compile(r"(?m)^\s*PART\s+(?:I{1,3}|IV)\s*$"),
    # a page number alone on its line
    re.compile(r"(?m)^\s*\d{1,4}\s*$"),
]


def clean_section(text: str) -> str:
    """Strip page furniture, then normalize whitespace."""
    for pattern in BOILERPLATE:
        text = pattern.sub(" ", text)

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
#
# Chunk boundaries land on sentence ends so no chunk starts mid-thought. The
# negative lookbehinds stop the splitter from breaking on abbreviations that
# are everywhere in filings: "U.S.", "Inc.", "No. 5", "e.g.".
# ---------------------------------------------------------------------------

_ABBREV = (
    r"(?<!\bU\.S)(?<!\bInc)(?<!\bCorp)(?<!\bLtd)(?<!\bCo)(?<!\bLLC)(?<!\bNo)"
    r"(?<!\bvs)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bJr)(?<!\bSr)(?<!\bSt)"
    r"(?<!\bi\.e)(?<!\be\.g)(?<!\bApprox)(?<!\bFig)(?<!\bNos)"
)
SENTENCE_END = re.compile(_ABBREV + r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9$])")


def split_sentences(text: str) -> list[str]:
    """Paragraph first, then sentence. Paragraph breaks are real boundaries in
    a filing (a risk factor heading, a new disclosure) so they are kept."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for sent in SENTENCE_END.split(para):
            sent = sent.strip()
            if sent:
                out.append(sent)
    return out


# ---------------------------------------------------------------------------
# Packing: sentences -> token windows
# ---------------------------------------------------------------------------

def _hard_split(text: str, tok: Tokenizer, budget: int, overlap: int) -> list[tuple[str, int]]:
    """Split one oversized run of text on exact token boundaries.

    This is the Item 8 case: a flattened financial table is thousands of
    tokens of numbers with no sentence punctuation anywhere, so the sentence
    packer has nothing to cut on. Cutting by token offsets keeps every window
    inside the budget instead of handing the embedder a 4,000-token blob.
    """
    offsets = [o for o in tok.offsets(text) if o[1] > o[0]]
    if not offsets:
        return []

    step = max(1, budget - overlap)
    pieces: list[tuple[str, int]] = []
    for start in range(0, len(offsets), step):
        window = offsets[start : start + budget]
        if not window:
            break
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            pieces.append((piece, len(window)))
        if start + budget >= len(offsets):
            break
    return pieces


def _tail_tokens(text: str, tok: Tokenizer, n_tokens: int) -> tuple[str, int]:
    """Last `n_tokens` tokens of `text`, cut on a token boundary."""
    offsets = [o for o in tok.offsets(text) if o[1] > o[0]]
    if not offsets:
        return "", 0
    window = offsets[-n_tokens:]
    return text[window[0][0] :].strip(), len(window)


def pack(units: list[tuple[str, int]], budget: int, overlap: int,
         tok: Tokenizer | None = None) -> list[str]:
    """Greedily fill windows up to `budget` tokens, carrying `overlap` tokens
    of trailing text into the next window.

    The overlap exists so a fact split across a boundary survives intact in at
    least one window. It costs roughly overlap/budget extra vectors: 64/512 is
    about 12% more chunks to embed and store.

    Carry-back prefers WHOLE sentences so a chunk does not open mid-thought.
    When the previous window ends in one very long sentence, no whole sentence
    fits in 64 tokens, and a strict rule would hand back zero overlap exactly
    where the boundary risk is highest. So the fallback carries the trailing
    tokens of that sentence instead: a slightly ragged opening beats a fact
    that exists in neither neighbouring chunk.
    """
    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_tokens = 0

    def flush() -> None:
        if cur:
            chunks.append(" ".join(t for t, _ in cur))

    for text, n in units:
        if cur and cur_tokens + n > budget:
            flush()

            tail: list[tuple[str, int]] = []
            tail_tokens = 0
            for sent, cnt in reversed(cur):
                if tail_tokens + cnt > overlap:
                    break
                tail.insert(0, (sent, cnt))
                tail_tokens += cnt

            if not tail and tok is not None:            # long-sentence fallback
                piece, cnt = _tail_tokens(cur[-1][0], tok, overlap)
                if piece:
                    tail, tail_tokens = [(piece, cnt)], cnt

            # guard: the carry-back must never make the new window overflow on
            # its first sentence, or packing cannot make progress
            if tail_tokens + n > budget:
                tail, tail_tokens = [], 0

            cur, cur_tokens = tail, tail_tokens

        cur.append((text, n))
        cur_tokens += n

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Registrant's Common Equity",
    "6": "Reserved", "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation", "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

NARRATIVE_ITEMS = {"1", "1A", "3", "5", "7", "7A"}   # the high-value text for RAG


@dataclass
class Chunk:
    chunk_id: str            # deterministic; the Qdrant point id
    tenant: str
    document_id: str         # cik/accession, the Postgres document key
    source_uri: str
    cik: str
    ticker: str | None
    company: str | None
    form_type: str
    filing_date: str | None
    fiscal_year: str | None
    item: str
    item_title: str
    is_narrative: bool
    chunk_index: int         # position in the whole document
    section_chunk_index: int # position within the item
    text: str                # clean text, this is what a citation quotes
    context_header: str      # prepended only for embedding, never shown
    token_count: int         # tokens in text (header excluded)
    char_count: int
    content_sha256: str
    embedding_model: str
    chunk_config: dict

    @property
    def embed_text(self) -> str:
        """What actually goes to the embedding server.

        A bare chunk from the middle of Item 1A reads as anonymous prose: no
        company, no year, no section. The header restores that so a query like
        "Apple supply chain risk 2025" can match on more than topic alone. The
        stored `text` stays clean so citations quote the filing, not our header.
        """
        return f"{self.context_header}\n\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embed_text"] = self.embed_text
        return d


def config_fingerprint() -> str:
    """Short hash of everything that changes what a chunk contains. It goes
    into every chunk_id so a re-chunk under new settings cannot collide with
    vectors built under the old ones."""
    payload = json.dumps(
        {
            "v": CHUNKER_VERSION,
            "strategy": CHUNK_STRATEGY,
            "size": CHUNK_SIZE_TOKENS,
            "overlap": CHUNK_OVERLAP_TOKENS,
            "min": MIN_CHUNK_TOKENS,
            "model": EMBEDDING_MODEL,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def meta_from_uri(uri: str) -> dict:
    """Recover cik / accession / document from the GCS layout written by
    edgar_downloader.py:  filings/{cik}/{accession_nodashes}/{document}"""
    parts = [p for p in re.sub(r"^gs://[^/]+/", "", uri).split("/") if p]
    meta = {"cik": None, "accession": None, "document": parts[-1] if parts else None}
    if len(parts) >= 3 and parts[-3].isdigit():
        meta["cik"] = parts[-3]
        meta["accession"] = parts[-2]
    return meta


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def chunk_sections(
    sections: dict[str, str],
    meta: dict,
    tok: Tokenizer,
    size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> tuple[list[Chunk], dict]:
    """{item: text} -> [Chunk]. Sections are chunked independently, so no
    chunk ever spans two items and every chunk has one unambiguous citation."""

    fingerprint = config_fingerprint()
    cik = str(meta.get("cik") or "unknown")
    accession = str(meta.get("accession") or "unknown")
    document_id = f"{cik}/{accession}"
    source_uri = meta.get("source_uri", "")

    chunks: list[Chunk] = []
    stats = {"dropped_short": 0, "hard_split": 0, "duplicates": 0, "per_item": {}}
    seen_hashes: set[str] = set()
    running_index = 0

    for item, raw in sections.items():
        text = clean_section(raw)
        title = ITEM_TITLES.get(item, "")

        header = (
            f"{meta.get('company') or cik} "
            f"({meta.get('ticker') or cik}) "
            f"{meta.get('form_type', '10-K')} "
            f"{meta.get('fiscal_year') or meta.get('filing_date') or ''} "
            f"- Item {item}. {title}"
        ).strip()
        # The header is embedded with the chunk, so its tokens come out of the
        # same 512 budget. Three costs are easy to forget and all three are
        # real: the header itself, the "\n\n" that joins it to the body, and
        # the [CLS]/[SEP] the encoder adds. Miss them and every chunk lands a
        # few tokens over the ceiling.
        budget = size - tok.count(f"{header}\n\n") - SPECIAL_TOKENS - SAFETY_TOKENS
        if budget <= max(min_tokens, overlap):
            raise ValueError(
                f"chunk budget {budget} (size {size} minus header/specials) is not "
                f"above min_chunk_tokens={min_tokens} / overlap={overlap} — "
                f"fix CHUNK_SIZE_TOKENS before ingesting")

        sentences = split_sentences(text)
        counts = tok.count_batch(sentences)

        units: list[tuple[str, int]] = []
        for sent, n in zip(sentences, counts):
            if n <= budget:
                units.append((sent, n))
            else:
                pieces = _hard_split(sent, tok, budget, overlap)
                stats["hard_split"] += len(pieces)
                units.extend(pieces)

        # Sentence counts do not simply add up once the sentences are joined:
        # the tokenizer merges across the join. Re-measure the finished windows
        # in one batch so token_count is the truth, not an estimate.
        windows = pack(units, budget, overlap, tok)
        window_tokens = tok.count_batch(windows)

        kept = 0
        for body, n_tokens in zip(windows, window_tokens):
            if n_tokens < min_tokens:
                stats["dropped_short"] += 1          # "None." / "[Reserved]"
                continue

            digest = hashlib.sha256(
                re.sub(r"\s+", " ", body).strip().lower().encode()
            ).hexdigest()
            if digest in seen_hashes:
                stats["duplicates"] += 1             # repeated legal language
                continue
            seen_hashes.add(digest)

            tenant = meta.get("tenant", DEFAULT_TENANT)
            name = f"{tenant}|{source_uri}#item={item}&i={kept}&cfg={fingerprint}"
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, name)),
                    tenant=meta.get("tenant", DEFAULT_TENANT),
                    document_id=document_id,
                    source_uri=source_uri,
                    cik=cik,
                    ticker=meta.get("ticker"),
                    company=meta.get("company"),
                    form_type=meta.get("form_type", "10-K"),
                    filing_date=meta.get("filing_date"),
                    fiscal_year=meta.get("fiscal_year"),
                    item=item,
                    item_title=title,
                    is_narrative=item in NARRATIVE_ITEMS,
                    chunk_index=running_index,
                    section_chunk_index=kept,
                    text=body,
                    context_header=header,
                    token_count=n_tokens,
                    char_count=len(body),
                    content_sha256=digest,
                    embedding_model=EMBEDDING_MODEL,
                    chunk_config={
                        "strategy": CHUNK_STRATEGY,
                        "size_tokens": size,
                        "overlap_tokens": overlap,
                        "min_tokens": min_tokens,
                        "version": CHUNKER_VERSION,
                        "fingerprint": fingerprint,
                    },
                )
            )
            kept += 1
            running_index += 1

        stats["per_item"][item] = {
            "chunks": kept,
            "tokens": sum(window_tokens),
            "chars": len(text),
        }

    return chunks, stats


def chunk_filing(uri: str, meta: dict | None = None, tok: Tokenizer | None = None,
                 items: list[str] | None = None) -> tuple[list[Chunk], dict]:
    """End to end for one filing: load -> text -> sections -> chunks."""
    meta = dict(meta or {})
    meta.setdefault("source_uri", uri)
    for key, value in meta_from_uri(uri).items():
        if value and not meta.get(key):
            meta[key] = value

    html = load_html(uri)
    text = html_to_text(html)
    sections = split_into_items(text)
    if items:
        sections = {k: v for k, v in sections.items() if k in items}

    tok = tok or Tokenizer()
    chunks, stats = chunk_sections(sections, meta, tok)
    stats["html_chars"] = len(html)
    stats["text_chars"] = len(text)
    stats["sections"] = len(sections)
    return chunks, stats


def write_jsonl(chunks: list[Chunk], path: str | Path) -> Path:
    """One JSON object per line. This is the handoff to the embedding stage:
    streamable, appendable, and resumable by line number."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _report(chunks: list[Chunk], stats: dict, sample: int) -> None:
    print(f"HTML {stats['html_chars']:,} chars -> text {stats['text_chars']:,} chars "
          f"-> {stats['sections']} sections -> {len(chunks):,} chunks\n")

    print(f"{'item':<5} {'chunks':>7} {'tokens':>9} {'avg':>6}  title")
    print("-" * 72)
    for item, s in stats["per_item"].items():
        avg = s["tokens"] // s["chunks"] if s["chunks"] else 0
        print(f"{item:<5} {s['chunks']:>7} {s['tokens']:>9,} {avg:>6}  {ITEM_TITLES.get(item, '')[:38]}")
    print("-" * 72)

    total_tokens = sum(c.token_count for c in chunks)
    over = [c for c in chunks if c.token_count > CHUNK_SIZE_TOKENS]
    print(f"{'ALL':<5} {len(chunks):>7} {total_tokens:>9,} "
          f"{total_tokens // max(1, len(chunks)):>6}")
    print(f"\ndropped (< {MIN_CHUNK_TOKENS} tokens): {stats['dropped_short']}   "
          f"exact-duplicate chunks: {stats['duplicates']}   "
          f"table windows hard-split: {stats['hard_split']}")
    print(f"over budget ({CHUNK_SIZE_TOKENS} tokens): {len(over)}   "
          f"<- must be 0, or the embedder truncates silently")

    # what the backfill actually costs, extrapolated from this one filing
    if chunks:
        print(f"\nper filing: {len(chunks):,} chunks | "
              f"5,000 filings: ~{len(chunks) * 5000:,} chunks "
              f"| ~{len(chunks) * 5000 * 1024 * 4 / 1e9:.1f} GB of raw f32 vectors")

    for chunk in chunks[:sample]:
        print("\n" + "=" * 72)
        print(f"chunk_id {chunk.chunk_id}  item {chunk.item}  "
              f"#{chunk.chunk_index}  {chunk.token_count} tokens")
        print(f"header:  {chunk.context_header}")
        print("-" * 72)
        print(chunk.text[:600] + ("..." if len(chunk.text) > 600 else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk a 10-K into embedding-sized pieces")
    ap.add_argument("--uri", default="gs://llm-platform-services-bucket/filings/320193/"
                                     "000032019325000079/aapl-20250927.htm",
                    help="gs:// URI or a local .htm path")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--company", default="Apple Inc.")
    ap.add_argument("--filing-date", default="2025-10-31")
    ap.add_argument("--fiscal-year", default="2025")
    ap.add_argument("--items", default="", help="comma-separated subset, e.g. 1A,7")
    ap.add_argument("--out", default="", help="output .jsonl (default: data/chunks/<doc>.jsonl)")
    ap.add_argument("--sample", type=int, default=1, help="chunks to print in full")
    args = ap.parse_args()

    meta = {
        "ticker": args.ticker,
        "company": args.company,
        "filing_date": args.filing_date,
        "fiscal_year": args.fiscal_year,
        "form_type": "10-K",
    }
    items = [i.strip() for i in args.items.split(",") if i.strip()] or None

    chunks, stats = chunk_filing(args.uri, meta, items=items)
    _report(chunks, stats, args.sample)

    stem = Path(meta_from_uri(args.uri)["document"] or "filing").stem
    out = Path(args.out) if args.out else Path("data/chunks") / f"{stem}.jsonl"
    write_jsonl(chunks, out)
    print(f"\nwrote {len(chunks):,} chunks -> {out}")


if __name__ == "__main__":
    main()
