"""Section-splitting DP — the TOC, the body, and the cross-reference trap."""

from app.ingestion.parser import split_into_items


def build_filing():
    """A synthetic 10-K with the three classic traps:
    - a TOC listing every item (full in-order chain, early)
    - real section headers (full in-order chain, spanning the doc)
    - a LATE cross-reference to Item 1A inside Item 7 (the NVIDIA bug)
    """
    toc = ("Item 1. Business 4 Item 1A. Risk Factors 12 Item 2. Properties 30 "
           "Item 3. Legal Proceedings 31 Item 7. Management's Discussion 35 ")
    body = (
        "Item 1. Business We make widgets. " + "The widget market is large. " * 30 +
        "Item 1A. Risk Factors Our business faces risks. " + "Risks abound here. " * 60 +
        "Item 2. Properties We own a factory. " + "It is a nice factory. " * 20 +
        "Item 3. Legal Proceedings None pending. " + "Courts are quiet. " * 10 +
        "Item 7. Management's Discussion and Analysis. Results were good. "
        'Read this in conjunction with "Item 1A. Risk Factors" of this report. ' +
        "Revenue grew nicely. " * 40
    )
    return toc + body


def test_body_headers_beat_toc():
    sections = split_into_items(build_filing())
    assert "1A" in sections and "7" in sections
    # 1A must hold the real risk text, not the one-line TOC entry
    assert "Risks abound" in sections["1A"]
    assert len(sections["1A"]) > 500


def test_late_cross_reference_does_not_steal_the_section():
    """The 'Refer to Item 1A' inside Item 7 comes AFTER Item 7's start; a
    last-match heuristic made it the section boundary (NVIDIA: 1 risk chunk).
    The in-order DP must reject it."""
    sections = split_into_items(build_filing())
    assert "Revenue grew" in sections["7"]
    assert "Risk Factors" not in sections["7"][:60] or "conjunction" not in sections["1A"]
    # 1A ends where Item 2 begins — the cross-reference stayed inside 7
    assert "conjunction" not in sections["1A"]


def test_flexible_separators():
    text = ("Item 1: Business " + "We sell things. " * 30 +
            "Item 1A — Risk Factors " + "Many risks. " * 30 +
            "Item 2. Properties " + "One office. " * 30)
    sections = split_into_items(text)
    assert set(sections) == {"1", "1A", "2"}


def test_empty_text():
    assert split_into_items("") == {}
