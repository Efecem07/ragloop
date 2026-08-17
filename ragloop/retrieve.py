"""
Hybrid retrieval — BM25 and dense vectors, fused by rank rather than by score.

WHY HYBRID. The two retrievers fail in opposite directions, which is the whole
argument for running both:

  dense misses literals    "error TS2345", "§ 5a", an order number, a rare product
                           name. The embedding of a rare token carries almost no
                           signal, so the nearest neighbours are merely on-topic.
  lexical misses paraphrase "my card was declined" against a document that says
                           "payment authorisation failed" shares no terms at all.

WHY RECIPROCAL RANK FUSION, NOT A WEIGHTED SUM. BM25 returns unbounded scores
whose scale depends on corpus statistics; cosine returns roughly 0 to 1. Adding
them requires normalising two distributions that shift every time the corpus
changes, and the weight you tuned on Tuesday is wrong on Friday. RRF throws the
scores away and keeps only the ranks:

    score(d) = sum over retrievers of  1 / (rrf_k + rank(d))

It has one parameter, it cannot be broken by a scale change, and a document that
both retrievers rank moderately well beats a document that only one of them
loves. That last property is the one that matters in practice.
"""
from __future__ import annotations

from typing import Sequence

from .bm25 import BM25
from .chunking import Chunk
from .embed import Embedder, Vector
from .store import Hit, InMemoryStore, VectorStore

# 60 is the value from the original RRF paper (Cormack et al., 2009). It damps
# the difference between the top ranks so that rank 1 and rank 3 are close and
# rank 1 and rank 50 are far, which is the shape you want when fusing rankers of
# unequal reliability.
RRF_K = 60


class HybridIndex:
    """A BM25 index and a vector store over the same chunks, fused with RRF.

    `embedder` is any callable `(list[str]) -> list[list[float]]`. Pass
    `hashing_embedder()` for offline tests, `openai_embedder()` in production.
    Set it to None to run lexical-only, which is a legitimate configuration for
    corpora that are mostly identifiers.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        store: VectorStore | None = None,
        rrf_k: int = RRF_K,
    ):
        self.embedder = embedder
        self.store: VectorStore = store if store is not None else InMemoryStore()
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._bm25: BM25 | None = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks. BM25 is rebuilt because its IDF depends on the whole
        corpus: adding documents changes the score of every existing one, and an
        index that pretends otherwise drifts quietly."""
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._bm25 = BM25([c.text for c in self.chunks])
        if self.embedder is not None:
            vectors: list[Vector] = self.embedder([c.text for c in chunks])
            self.store.add(chunks, vectors)

    def search(self, query: str, k: int = 5, *, candidates: int = 30) -> list[Hit]:
        """Return the top k chunks for a query.

        `candidates` is how deep each retriever is asked to go before fusion.
        Fusing only the top k from each would throw away exactly the documents
        RRF exists to rescue: the ones ranked 8th by both retrievers rather than
        1st by one of them.
        """
        if not self.chunks:
            return []
        depth = max(candidates, k)

        ranks: dict[int, dict[str, int]] = {}

        # Ranks are 1-based to match the formula as published: the best document
        # contributes 1/(k+1), not 1/k.
        if self._bm25 is not None:
            for rank, (doc_i, _score) in enumerate(self._bm25.search(query, depth), start=1):
                ranks.setdefault(doc_i, {})["bm25"] = rank

        if self.embedder is not None and len(self.store):
            qvec = self.embedder([query])[0]
            for rank, (doc_i, _score) in enumerate(self.store.search(qvec, depth), start=1):
                ranks.setdefault(doc_i, {})["dense"] = rank

        fused: list[Hit] = []
        for doc_i, per_retriever in ranks.items():
            contributions = {
                name: 1.0 / (self.rrf_k + rank) for name, rank in per_retriever.items()
            }
            fused.append(
                Hit(
                    chunk=self.chunks[doc_i],
                    score=sum(contributions.values()),
                    source_scores=contributions,
                )
            )

        # Ties are broken by chunk order so results are reproducible run to run.
        fused.sort(key=lambda h: (-h.score, h.chunk.id))
        return fused[:k]

    def __len__(self) -> int:
        return len(self.chunks)


def build_context(hits: Sequence[Hit], *, max_chars: int = 4000) -> str:
    """Format hits as a prompt context block with citable sources.

    Truncation happens on a chunk boundary, never mid-chunk: half a chunk in the
    context is a fact the model can misread with full confidence.
    """
    parts: list[str] = []
    used = 0
    for hit in hits:
        block = f"[{hit.chunk.id}]\n{hit.chunk.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)
