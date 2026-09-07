"""
parser.py

Parse a 10-K filing that's already in GCS:
  load HTML from GCS -> clean text -> split into Item sections -> summarize
"""

import re
import warnings
from pathlib import Path
from urllib.parse import urlparse

# A 10-K .htm is really iXBRL (XHTML + inline XBRL), so bs4 warns that we are
# parsing XML with an HTML parser. Extraction is correct either way; the warning
# is noise once you have confirmed the section split.
try:
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:      # bs4 not installed yet
    pass


def load_html_from_gcs(gcs_uri: str) -> str:
    """Download an HTML filing from a gs:// URI and return it as a string."""
    from google.cloud import storage

    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()
    return data.decode("utf-8", errors="replace")


def load_html(source: str) -> str:
    """Load a filing from a gs:// URI or a local path.

    The local branch is what makes this pipeline testable offline: download one
    filing once, then iterate on chunking without touching GCS on every run.
    """
    if source.startswith("gs://"):
        return load_html_from_gcs(source)
    return Path(source).read_text(encoding="utf-8", errors="replace")


# Tags that end a line of text. Everything else is inline: EDGAR filings wrap
# single WORDS in nested <span>s for styling, so a global separator=" " split
# words mid-letter ("RISK" -> "RIS K" in Microsoft's 10-K, which then broke
# section matching). Inline content must join seamlessly; blocks get newlines,
# table cells get spaces.
_BLOCK_TAGS = ["p", "div", "br", "tr", "table", "li", "ul", "ol",
               "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "hr"]
_CELL_TAGS = ["td", "th"]


def html_to_text(html: str) -> str:
    """Strip HTML (and inline XBRL) down to readable text, preserving words
    across styled inline spans and boundaries across real blocks."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")   # if lxml won't install, use "html.parser"

    for tag in soup(["script", "style"]):      # drop non-content
        tag.decompose()

    for tag in soup.find_all(_CELL_TAGS):      # cells: space, not newline
        tag.insert_after(" ")
    for tag in soup.find_all(_BLOCK_TAGS):     # blocks: end the line
        tag.insert_after("\n")

    text = soup.get_text()                     # NO separator: words stay whole

    text = text.replace("\xa0", " ")           # non-breaking spaces
    text = re.sub(r"[ \t]+", " ", text)         # collapse runs of spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)    # collapse blank lines
    return text.strip()


# Ordered 10-K items. The number+period is the anchor; the loose title keyword
# just helps avoid matching stray "see Item 1A" cross-references in body text.
# Ordered 10-K items. `_S` is the separator companies actually use after the
# item number: "Item 1A. Risk Factors", "Item 1A: ...", "Item 1A — ...".
_S = r"\s*[\.\:\u2013\u2014-]?\s*"
ITEMS = [
    ("1",  rf"Item\s+1{_S}Business"),
    ("1A", rf"Item\s+1A{_S}Risk\s+Factors"),
    ("1B", rf"Item\s+1B{_S}Unresolved\s+Staff\s+Comments"),
    ("1C", rf"Item\s+1C{_S}Cybersecurity"),
    ("2",  rf"Item\s+2{_S}Propert"),
    ("3",  rf"Item\s+3{_S}Legal\s+Proceedings"),
    ("4",  rf"Item\s+4{_S}Mine\s+Safety"),
    ("5",  rf"Item\s+5{_S}Market\s+for"),
    ("6",  rf"Item\s+6{_S}(?:\[?\s*Reserved|Selected)"),
    ("7",  rf"Item\s+7{_S}Management"),
    ("7A", rf"Item\s+7A{_S}Quantitative"),
    ("8",  rf"Item\s+8{_S}Financial\s+Statements"),
    ("9",  rf"Item\s+9{_S}Changes"),
    ("9A", rf"Item\s+9A{_S}Controls"),
    ("9B", rf"Item\s+9B{_S}Other\s+Information"),
    ("10", rf"Item\s+10{_S}Directors"),
    ("11", rf"Item\s+11{_S}Executive\s+Compensation"),
    ("12", rf"Item\s+12{_S}Security\s+Ownership"),
    ("13", rf"Item\s+13{_S}Certain\s+Relationships"),
    ("14", rf"Item\s+14{_S}Principal\s+Accountant"),
    ("15", rf"Item\s+15{_S}Exhibit"),
]



def split_into_items(text: str) -> dict:
    """Split filing text into {item_id: section_text}.

    Every Item title appears many times in a 10-K: the table of contents, the
    real section header, and in-body cross-references ("Refer to Item 1A...").
    The old take-the-LAST-match heuristic broke on both directions — NVIDIA's
    late cross-references pushed Risk Factors to the end of the document, and
    exhibit-index recurrences shifted boundaries (review finding #32).

    Now: collect ALL matches per item and pick ONE per item so that positions
    are strictly increasing in the canonical item order (a 10-K always
    presents items in order), maximizing (items placed, then total position)
    — the "then total position" tie-break is what beats the table of
    contents, which is also a full in-order chain but sits earlier in the
    document than the real sections.  O(n²) DP over a few hundred matches.
    """
    events: list[tuple[int, int, str]] = []          # (pos, item_rank, item_id)
    for rank, (item_id, pattern) in enumerate(ITEMS):
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            events.append((m.start(), rank, item_id))
    if not events:
        return {}
    events.sort()

    n = len(events)
    # dp[i] = (chain_len, pos_sum, prev_index) for the best chain ending at i
    dp: list[tuple[int, int, int]] = [(1, events[i][0], -1) for i in range(n)]
    for i in range(n):
        for j in range(i):
            if events[j][1] < events[i][1]:          # strictly later item
                cand = (dp[j][0] + 1, dp[j][1] + events[i][0], j)
                if (cand[0], cand[1]) > (dp[i][0], dp[i][1]):
                    dp[i] = cand

    best = max(range(n), key=lambda i: (dp[i][0], dp[i][1]))
    chain: list[tuple[int, str]] = []
    i = best
    while i != -1:
        chain.append((events[i][0], events[i][2]))
        i = dp[i][2]
    chain.reverse()

    sections: dict[str, str] = {}
    for k, (pos, item_id) in enumerate(chain):
        end = chain[k + 1][0] if k + 1 < len(chain) else len(text)
        sections[item_id] = text[pos:end].strip()
    return sections


if __name__ == "__main__":
    GCS_URI = "gs://llm-platform-services-bucket/filings/320193/000032019325000079/aapl-20250927.htm"

    html = load_html_from_gcs(GCS_URI)
    print(f"Loaded HTML: {len(html):,} chars")

    text = html_to_text(html)
    print(f"Clean text: {len(text):,} chars\n")

    sections = split_into_items(text)
    print(f"Found {len(sections)} sections:\n")
    for item_id, body in sections.items():
        preview = body[:90].replace("\n", " ")
        print(f"  Item {item_id:<3} {len(body):>8,} chars  |  {preview}...")