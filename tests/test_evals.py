"""Metrics are checked against hand-computed values. An eval harness whose own
arithmetic is untested is worse than no harness: it produces numbers people act
on."""
from __future__ import annotations

import math

import pytest

from ragloop import Chunk, HybridIndex, QueryCase, ablation, evaluate, markdown_ablation

DOCS = [
    Chunk("Password resets happen in the admin panel under Security.", "help.md", 0),
    Chunk("A reset mail arrives within five minutes of the request.", "help.md", 1),
    Chunk("Invoices are issued on the first of each month.", "billing.md", 0),
    Chunk("The office is open from nine to five.", "office.md", 0),
]


def index_of(chunks=DOCS) -> HybridIndex:
    idx = HybridIndex(embedder=None)
    idx.add(chunks)
    return idx


def test_recall_counts_all_relevant_chunks_not_just_the_first():
    """A question needing two documents is only half answered by one of them.
    The lenient 'at least one hit' reading would score this 1.0 and hide it."""
    idx = index_of()
    case = QueryCase("password reset", {"help.md#0", "help.md#1"})
    m = evaluate(idx, [case], k=1)
    assert m.recall_at_k == pytest.approx(0.5)


def test_perfect_retrieval_scores_one_across_the_board():
    idx = index_of()
    case = QueryCase("password reset", {"help.md#0", "help.md#1"})
    m = evaluate(idx, [case], k=4)
    assert m.recall_at_k == pytest.approx(1.0)
    assert m.mrr == pytest.approx(1.0)
    assert m.misses == []


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank():
    idx = index_of()
    case = QueryCase("office invoices", {"billing.md#0"})
    m = evaluate(idx, [case], k=4)
    ranked = [h.chunk.id for h in idx.search(case.query, k=4)]
    expected_rank = ranked.index("billing.md#0") + 1
    assert m.mrr == pytest.approx(1 / expected_rank)


def test_a_query_with_no_relevant_hit_is_recorded_by_name():
    idx = index_of()
    case = QueryCase("kubernetes autoscaling", {"help.md#0"})
    m = evaluate(idx, [case], k=3)
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0
    assert m.misses == ["kubernetes autoscaling"]


def test_ndcg_discounts_by_position():
    idx = index_of()
    top = evaluate(idx, [QueryCase("password resets admin panel", {"help.md#0"})], k=4)
    # One relevant chunk at rank 1 gives nDCG 1.0; anywhere lower is strictly less.
    assert 0.0 < top.ndcg_at_k <= 1.0
    if top.mrr == pytest.approx(1.0):
        assert top.ndcg_at_k == pytest.approx(1.0)
    else:
        assert top.ndcg_at_k < 1.0


def test_ndcg_matches_the_formula_for_a_known_ranking():
    idx = index_of()
    case = QueryCase("password resets admin panel security", {"help.md#0"})
    m = evaluate(idx, [case], k=4)
    hits = [h.chunk.id for h in idx.search(case.query, k=4)]
    rank = hits.index("help.md#0")
    expected = (1 / math.log2(rank + 2)) / 1.0
    assert m.ndcg_at_k == pytest.approx(expected)


def test_evaluating_zero_queries_raises_instead_of_reporting_success():
    """An empty eval set averaging to 1.0 is the most dangerous number a harness
    can print, because it looks like a pass."""
    with pytest.raises(ValueError):
        evaluate(index_of(), [], k=5)


def test_ablation_reports_each_retriever_separately():
    def stub(texts):
        return [[1.0, 0.0] if "reset" in t.lower() else [0.0, 1.0] for t in texts]

    idx = HybridIndex(embedder=stub)
    idx.add(DOCS)
    cases = [QueryCase("password reset", {"help.md#0", "help.md#1"})]
    results = ablation(idx, cases, k=2)
    assert set(results) == {"lexical", "dense", "hybrid"}
    assert all(0.0 <= m.recall_at_k <= 1.0 for m in results.values())


def test_ablation_without_an_embedder_omits_the_dense_row():
    results = ablation(index_of(), [QueryCase("password reset", {"help.md#0"})], k=2)
    assert set(results) == {"lexical", "hybrid"}


def test_ablation_does_not_mutate_the_index_it_measures():
    idx = index_of()
    before = len(idx.chunks)
    ablation(idx, [QueryCase("password reset", {"help.md#0"})], k=2)
    assert len(idx.chunks) == before
    assert idx._bm25 is not None


def test_markdown_table_is_well_formed():
    results = ablation(index_of(), [QueryCase("password reset", {"help.md#0"})], k=2)
    md = markdown_ablation(results)
    rows = [ln for ln in md.splitlines() if ln.startswith("| ")]
    assert len(rows) == 1 + len(results)  # header plus one row per retriever
    assert all(row.count("|") == 6 for row in rows)  # five columns
    assert "|---|---|---|---|---|" in md
