"""BM25 is here for the queries embeddings get wrong, so the tests are about
literals, rarity and length."""
from __future__ import annotations

from ragloop import BM25, tokenize

DOCS = [
    "The payment failed with error code TS2345 during checkout.",
    "Checkout is the last step of the ordering process.",
    "Payments are settled once a day by the provider.",
    "A very long document about checkout and payment and ordering and delivery "
    "and invoices and refunds and customers and accounts and settings and more "
    "checkout and more payment, going on at length without ever mentioning the "
    "specific error code the user is asking about.",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Payment IS in the Checkout") == ["payment", "checkout"]


def test_tokenize_keeps_identifiers_intact():
    # The whole reason BM25 is in this library.
    assert "ts2345" in tokenize("error code TS2345 appeared")


def test_exact_identifier_beats_topical_similarity():
    bm25 = BM25(DOCS)
    top = bm25.search("TS2345", k=1)
    assert top[0][0] == 0


def test_rare_terms_outweigh_common_ones():
    bm25 = BM25(DOCS)
    # "checkout" appears in three documents, "TS2345" in one. A query with both
    # must be dominated by the rare term.
    assert bm25.idf["ts2345"] > bm25.idf["checkout"]


def test_length_normalisation_stops_the_padded_document_winning():
    bm25 = BM25(DOCS)
    ranked = bm25.search("checkout payment", k=4)
    best = ranked[0][0]
    # Document 3 repeats both terms most often but is mostly filler. Without
    # length normalisation it would win on raw term frequency alone.
    assert best != 3


def test_idf_is_never_negative_for_ubiquitous_terms():
    # A term in every document must not drag a matching doc below a
    # non-matching one, which is what an unsmoothed IDF would do.
    bm25 = BM25(["alpha term", "alpha other", "alpha third"])
    assert bm25.idf["alpha"] >= 0.0


def test_unknown_term_returns_nothing_rather_than_everything():
    bm25 = BM25(DOCS)
    assert bm25.search("kubernetes", k=5) == []


def test_empty_corpus_is_survivable():
    bm25 = BM25([])
    assert bm25.search("anything", k=3) == []


def test_corpus_of_only_stopwords_does_not_divide_by_zero():
    """Every document tokenising to nothing puts avgdl at zero, which is one
    division away from a crash on the first query."""
    bm25 = BM25(["", "the and of", "   "])
    assert bm25.avgdl == 0.0
    assert bm25.search("the", k=3) == []
    assert bm25.search("anything", k=3) == []


def test_one_empty_document_alongside_real_ones():
    bm25 = BM25(["", "checkout payment failed"])
    assert bm25.search("checkout", k=2)[0][0] == 1
