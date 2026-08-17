"""The point of hybrid retrieval is that each half rescues the other. These tests
construct exactly those two situations and assert the rescue happens."""
from __future__ import annotations

from ragloop import Chunk, HybridIndex, build_context, hashing_embedder

LEXICAL_ONLY = Chunk("Error TS2345 is raised when the argument type does not match.", "errors.md", 0)
PARAPHRASE = Chunk("Your card was declined by the issuing bank.", "billing.md", 0)
NOISE = [
    Chunk("The office is open from nine to five on weekdays.", "office.md", 0),
    Chunk("Team lunches happen every second Thursday.", "office.md", 1),
    Chunk("Parking spaces can be booked in the intranet.", "office.md", 2),
]


def semantic_stub():
    """A tiny embedder that knows one synonym pair and nothing else.

    A hashing embedder shares vocabulary and therefore cannot demonstrate the
    thing worth demonstrating: that the dense half finds a document with no terms
    in common with the query. So the pair is stubbed explicitly, which keeps the
    assertion honest about what is being tested.
    """
    payment_topic = [1.0, 0.0, 0.0]
    other_topic = [0.0, 1.0, 0.0]

    def embed(texts):
        out = []
        for t in texts:
            low = t.lower()
            if "declined" in low or "issuing bank" in low or "payment was rejected" in low:
                out.append(payment_topic)
            else:
                out.append(other_topic)
        return out

    return embed


def test_dense_half_finds_a_document_with_no_shared_terms():
    index = HybridIndex(embedder=semantic_stub())
    index.add([PARAPHRASE, *NOISE])
    hits = index.search("my payment was rejected", k=1)
    assert hits[0].chunk.id == "billing.md#0"
    assert "dense" in hits[0].source_scores


def test_lexical_half_finds_an_identifier_the_embedder_cannot_represent():
    index = HybridIndex(embedder=semantic_stub())
    index.add([LEXICAL_ONLY, *NOISE])
    hits = index.search("TS2345", k=1)
    assert hits[0].chunk.id == "errors.md#0"
    assert "bm25" in hits[0].source_scores


def test_a_document_both_retrievers_like_outranks_one_a_single_retriever_loves():
    """The property that justifies RRF over picking one retriever."""
    both = Chunk("The payment was declined and the card was rejected.", "both.md", 0)
    index = HybridIndex(embedder=semantic_stub())
    index.add([both, PARAPHRASE, *NOISE])
    hits = index.search("payment was rejected", k=2)
    assert hits[0].chunk.id == "both.md#0"
    assert set(hits[0].source_scores) == {"bm25", "dense"}


def test_lexical_only_configuration_is_supported():
    index = HybridIndex(embedder=None)
    index.add([LEXICAL_ONLY, *NOISE])
    hits = index.search("TS2345", k=1)
    assert hits[0].chunk.id == "errors.md#0"
    assert set(hits[0].source_scores) == {"bm25"}


def test_empty_index_returns_no_hits_rather_than_raising():
    assert HybridIndex(embedder=hashing_embedder()).search("anything") == []


def test_adding_nothing_is_a_no_op():
    index = HybridIndex(embedder=hashing_embedder())
    index.add([])
    assert len(index) == 0


def test_incremental_adds_rebuild_the_lexical_statistics():
    """BM25 IDF depends on the whole corpus. If a second add() did not rebuild,
    the new documents would be scored against stale statistics."""
    index = HybridIndex(embedder=None)
    index.add([LEXICAL_ONLY])
    index.add(NOISE)
    assert len(index) == 4
    assert index.search("TS2345", k=1)[0].chunk.id == "errors.md#0"


def test_results_are_deterministic_across_runs():
    index = HybridIndex(embedder=hashing_embedder())
    index.add([LEXICAL_ONLY, PARAPHRASE, *NOISE])
    first = [h.chunk.id for h in index.search("office parking", k=3)]
    second = [h.chunk.id for h in index.search("office parking", k=3)]
    assert first == second


def test_build_context_truncates_on_chunk_boundaries():
    index = HybridIndex(embedder=None)
    index.add([LEXICAL_ONLY, PARAPHRASE, *NOISE])
    hits = index.search("office", k=5)
    ctx = build_context(hits, max_chars=80)
    # Never a partial chunk: every chunk that appears, appears in full.
    for hit in hits:
        if hit.chunk.text[:20] in ctx:
            assert hit.chunk.text in ctx


def test_build_context_keeps_at_least_one_chunk_even_if_oversized():
    """Returning an empty context because the best chunk is long is worse than
    returning one oversized chunk: the caller has nothing to answer from."""
    long_chunk = Chunk("parking " * 100, "big.md", 0)
    index = HybridIndex(embedder=None)
    index.add([long_chunk])
    hits = index.search("parking", k=1)
    assert hits, "precondition: the chunk must be retrievable"
    ctx = build_context(hits, max_chars=50)
    assert "big.md" in ctx


def test_context_carries_citable_ids():
    index = HybridIndex(embedder=None)
    index.add([LEXICAL_ONLY])
    assert "[errors.md#0]" in build_context(index.search("TS2345", k=1))
