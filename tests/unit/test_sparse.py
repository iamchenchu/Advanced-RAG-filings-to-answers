"""Hand-written BM25 — the math must be right and deterministic."""

import math

from app.retrieval.sparse import (analyze, encode_document, index_weights,
                                  query_weights, term_id)


def test_analyzer_lowercases_and_drops_stopwords():
    assert analyze("The iPhone AND the Mac") == ["iphone", "mac"]


def test_term_ids_are_stable_and_31bit():
    assert term_id("revenue") == term_id("revenue")
    assert 0 <= term_id("revenue") < 2**31


def test_encode_document_counts_and_length():
    tf, terms, dl = encode_document("revenue revenue grew; revenue grew.")
    assert dl == 5                                   # 3x revenue + 2x grew
    by_term = dict(zip(terms, [tf[term_id(t)] for t in terms]))
    assert by_term == {"revenue": 3, "grew": 2}


def test_idf_prefers_rare_terms():
    dfs = {term_id("rare"): 1, term_id("common"): 90}
    w = query_weights("rare common", dfs, n_chunks=100)
    assert w[term_id("rare")] > w[term_id("common")] > 0


def test_unseen_terms_get_no_weight():
    w = query_weights("zzzunseen", {}, n_chunks=100)
    assert w == {}


def test_tf_saturation():
    """BM25's point vs raw tf: the 50th occurrence adds almost nothing."""
    w1 = index_weights({1: 1}, 100, 100.0, 1.2, 0.75)[1]
    w5 = index_weights({1: 5}, 100, 100.0, 1.2, 0.75)[1]
    w50 = index_weights({1: 50}, 100, 100.0, 1.2, 0.75)[1]
    assert w1 < w5 < w50
    assert (w50 - w5) < (w5 - w1)                   # diminishing returns
    assert w50 < 1.2 + 1                            # bounded by k1+1


def test_length_normalization_penalizes_long_docs():
    short = index_weights({1: 2}, 50, 100.0, 1.2, 0.75)[1]
    long_ = index_weights({1: 2}, 400, 100.0, 1.2, 0.75)[1]
    assert short > long_


def test_full_bm25_score_composes():
    """index weight x query idf == textbook BM25 for one term."""
    n, df, tf, dl, avgdl, k1, b = 1000, 10, 3, 120, 100.0, 1.2, 0.75
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    expected = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    got = (index_weights({1: tf}, dl, avgdl, k1, b)[1]
           * query_weights("x", {term_id("x"): df}, n)[term_id("x")])
    # same term id on both sides
    got = index_weights({term_id("x"): tf}, dl, avgdl, k1, b)[term_id("x")] * \
          query_weights("x", {term_id("x"): df}, n)[term_id("x")]
    assert abs(got - expected) < 1e-9
