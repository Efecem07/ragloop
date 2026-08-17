"""
Retrieval evaluation — the part that turns an opinion into a number.

"We added RAG" is not a claim anyone can check. "Recall@5 went from 0.62 to 0.87
and here is the ablation" is. This module is the reason the rest of the library
exists in this shape: every design choice in chunking and fusion is testable, so
none of them should be argued about.

WHY THESE METRICS

  Recall@k     did the answer make it into the context window at all? This is the
               ceiling on everything downstream. A generator cannot cite what it
               never saw, so a retrieval bug looks exactly like a hallucination
               and gets misdiagnosed as one.
  MRR          how high did the first relevant chunk land? Position matters:
               models attend unevenly across a long context, and the chunk at
               rank 1 gets read differently from the chunk at rank 9.
  nDCG@k       the graded version: rewards putting several relevant chunks near
               the top, discounted by position.

WHY THE ABLATION IS THE HEADLINE. Hybrid retrieval costs an embedding call and a
second index. `ablation()` runs lexical-only, dense-only and hybrid over the same
queries so that cost is either justified by numbers or dropped. Running it is
usually more informative than tuning anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .retrieve import HybridIndex


@dataclass
class QueryCase:
    """One evaluation query and the chunk ids that genuinely answer it.

    `relevant` holds Chunk.id values. Building this set is the real work of
    evaluating retrieval, and there is no way around it: a labelled set of thirty
    queries tells you more than any amount of looking at outputs.
    """

    query: str
    relevant: set[str]
    note: str = ""


@dataclass
class Metrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    n_queries: int
    misses: list[str] = field(default_factory=list)  # queries with nothing relevant retrieved

    def __str__(self) -> str:
        return (
            f"recall@{self.k}={self.recall_at_k:.3f}  "
            f"MRR={self.mrr:.3f}  nDCG@{self.k}={self.ndcg_at_k:.3f}  "
            f"({self.n_queries} queries, {len(self.misses)} with no hit)"
        )


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def evaluate(index: HybridIndex, cases: Sequence[QueryCase], k: int = 5) -> Metrics:
    """Score an index over labelled queries.

    Recall here is "fraction of relevant chunks retrieved", averaged per query,
    not "did we get at least one". The stricter reading is the useful one when a
    question needs two documents to answer it, and the lenient one hides exactly
    that failure.
    """
    if not cases:
        raise ValueError("no query cases: an eval over zero queries reports 1.0 and means nothing")

    recalls: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    misses: list[str] = []

    for case in cases:
        hits = index.search(case.query, k=k)
        retrieved = [h.chunk.id for h in hits]
        relevant = case.relevant

        found = [cid for cid in retrieved if cid in relevant]
        recalls.append(len(found) / len(relevant) if relevant else 0.0)
        if not found:
            misses.append(case.query)

        rr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        gains = [1.0 if cid in relevant else 0.0 for cid in retrieved]
        ideal = [1.0] * min(len(relevant), k)
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(gains) / idcg if idcg else 0.0)

    n = len(cases)
    return Metrics(
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
        ndcg_at_k=sum(ndcgs) / n,
        k=k,
        n_queries=n,
        misses=misses,
    )


def ablation(index: HybridIndex, cases: Sequence[QueryCase], k: int = 5) -> dict[str, Metrics]:
    """Score lexical-only, dense-only and hybrid over the same queries.

    The three indexes share the already-computed chunks and vectors; only the
    retrievers are switched off, so nothing is re-embedded and the comparison is
    on identical data.
    """
    results: dict[str, Metrics] = {}

    lexical = HybridIndex(embedder=None, rrf_k=index.rrf_k)
    lexical.chunks = index.chunks
    lexical._bm25 = index._bm25
    results["lexical"] = evaluate(lexical, cases, k=k)

    if index.embedder is not None and len(index.store):
        dense = HybridIndex(embedder=index.embedder, store=index.store, rrf_k=index.rrf_k)
        dense.chunks = index.chunks
        dense._bm25 = None
        results["dense"] = evaluate(dense, cases, k=k)

    results["hybrid"] = evaluate(index, cases, k=k)
    return results


def print_ablation(results: dict[str, Metrics], title: str = "RETRIEVAL ABLATION") -> None:
    print(f"\n=== {title} ===")
    for name, m in results.items():
        print(f"  {name:<9} {m}")
    if "hybrid" in results and "lexical" in results:
        delta = results["hybrid"].recall_at_k - results["lexical"].recall_at_k
        verdict = "worth it" if delta > 0.01 else "not paying for itself on this corpus"
        print(f"  hybrid vs lexical: recall {delta:+.3f}  ->  {verdict}")


def markdown_ablation(results: dict[str, Metrics], title: str = "Retrieval ablation") -> str:
    """Render as a Markdown table, for a README or a CI summary."""
    k = next(iter(results.values())).k
    lines = [
        f"**{title}**",
        "",
        f"| retriever | recall@{k} | MRR | nDCG@{k} | queries with no hit |",
        "|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m.recall_at_k:.3f} | {m.mrr:.3f} | {m.ndcg_at_k:.3f} | {len(m.misses)} |"
        )
    return "\n".join(lines)
