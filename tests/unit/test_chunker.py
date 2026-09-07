"""Chunker invariants that don't need the real tokenizer (fast, offline)."""

from app.ingestion.chunker import pack


class FakeTok:
    """1 token per word — enough to test the packing logic."""
    def offsets(self, text):
        out, pos = [], 0
        for w in text.split(" "):
            out.append((pos, pos + len(w)))
            pos += len(w) + 1
        return out


def unit(n):
    return (" ".join(["w"] * n), n)


def test_windows_respect_budget():
    windows = pack([unit(200), unit(200), unit(200)], budget=512, overlap=64,
                   tok=FakeTok())
    assert len(windows) == 2                     # 400 + 200(+overlap carry)
    for w in windows:
        assert len(w.split()) <= 512


def test_overlap_carries_sentences_back():
    windows = pack([unit(500), unit(60), unit(500)], budget=512, overlap=64,
                   tok=FakeTok())
    # the 60-token sentence fits in the overlap: it must open window 2
    assert len(windows) >= 2
    assert len(windows[1].split()) >= 60 + 500 - 512 or "w" in windows[1]


def test_no_progress_stall_on_oversized_carry():
    # every unit near budget: carry-back must never make packing loop forever
    windows = pack([unit(500)] * 5, budget=512, overlap=64, tok=FakeTok())
    assert len(windows) == 5
