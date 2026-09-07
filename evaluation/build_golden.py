"""
evaluation/build_golden.py — build the LABELED golden dataset.

The flat-collection eval (run_eval.py) measures ANN approximation error only.
THIS is the dataset that measures whether retrieval finds the RIGHT passage:
each entry is a natural-language query labeled with the chunk(s) that
actually contain its answer.

Labels are CONTENT-ANCHORED, not id-anchored: each spec names the ticker, the
10-K section, and distinctive answer phrases; this script resolves them to
concrete chunk ids against the live corpus. Re-chunk the corpus (new ids) and
the labels re-resolve — the golden set survives every chunking experiment.

    python evaluation/build_golden.py            # writes fixtures/golden.jsonl

Honest caveat: phrase-anchored labels slightly favor lexical retrieval when a
query shares words with its anchor. The queries below are paraphrased, not
copied, to keep that bias small.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.persistence.db import close_pool, get_pool  # noqa: E402

OUT = Path(__file__).parent / "fixtures" / "golden.jsonl"

# (query, ticker, section-or-None, [phrases every relevant chunk must contain])
SPEC = [
    ("What are Apple's main supply chain risks?",
     "AAPL", "1A", ["outsourcing partners"]),
    ("How does Apple describe competition in the smartphone market?",
     "AAPL", "1", ["Competition has been particularly intense"]),
    ("What lawsuits does Apple face over the App Store?",
     "AAPL", "3", ["Epic Games"]),
    ("How is the board involved in overseeing cyber threats at Apple?",
     "AAPL", "1C", ["Audit Committee"]),
    ("How much stock did Apple buy back and what dividends were declared?",
     "AAPL", "7", ["repurchase"]),
    ("Who is Apple's independent registered public accounting firm?",
     "AAPL", "8", ["Ernst & Young"]),
    ("What was Apple's revenue in its products and services segments?",
     "AAPL", "7", ["Products and Services Performance"]),
    ("What risks does Microsoft see in AI development and deployment?",
     "MSFT", "1A", ["artificial intelligence"]),
    ("Which reportable segments does Microsoft disclose?",
     "MSFT", "8", ["Intelligent Cloud"]),
    ("How do US export controls limit NVIDIA's sales to China?",
     "NVDA", "1A", ["export"]),
    ("What competition does Amazon Web Services face?",
     "AMZN", "1A", ["AWS"]),
    ("What does Tesla say about its self-driving capabilities?",
     "TSLA", "1", ["self-driving"]),
    ("How much has Meta invested in Reality Labs and the metaverse?",
     "META", "7", ["Reality Labs"]),
    ("What antitrust actions is Alphabet defending?",
     "GOOGL", "8", ["antitrust"]),   # Alphabet discloses these in the Item 8
                                     # contingencies note, not Item 3
    ("What credit risks does JPMorgan carry in its lending business?",
     "JPM", "1A", ["credit"]),
    ("How is Walmart growing its online business?",
     "WMT", "7", ["eCommerce"]),
    ("What talc-related litigation does Johnson & Johnson face?",
     "JNJ", None, ["talc"]),         # JNJ spreads talc across 1A/7/8
    ("What seasonality does Apple's business show?",
     "AAPL", "1", ["Seasonality"]),
]


async def main() -> None:
    pool = await get_pool()
    entries = []
    print(f"{'query':<58} {'tkr':<6} {'sec':<4} {'labeled'}")
    print("-" * 84)
    for query, ticker, section, phrases in SPEC:
        conditions = ["d.meta->>'ticker' = $1"]
        params: list = [ticker]
        if section:
            conditions.append(f"c.section = ${len(params) + 1}")
            params.append(section)
        for phrase in phrases:
            conditions.append(f"c.text ILIKE ${len(params) + 1}")
            params.append(f"%{phrase}%")
        rows = await pool.fetch(
            f"""SELECT c.id FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE {' AND '.join(conditions)}
                ORDER BY c.chunk_index LIMIT 60""",
            # the corpus holds ~10 years per company; every year's copy of the
            # answer passage is relevant, not just the first file's five
            *params)
        ids = [str(r["id"]) for r in rows]
        status = f"{len(ids)} chunk(s)" if ids else "!! UNRESOLVED"
        print(f"{query[:57]:<58} {ticker:<6} {section or '-':<4} {status}")
        if ids:
            entries.append({"query": query, "ticker": ticker, "section": section,
                            "anchor_phrases": phrases, "relevant_chunk_ids": ids})

    OUT.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    print(f"\n{len(entries)}/{len(SPEC)} entries resolved -> {OUT}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
