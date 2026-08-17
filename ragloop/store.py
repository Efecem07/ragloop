"""
Vector store — an interface with one obvious implementation.

`VectorStore` is deliberately four methods wide. A managed vector database
(Qdrant, pgvector, Chroma) implements the same four and drops in without the
retriever changing, which is the point of naming the seam at all.

The bundled `InMemoryStore` does exact search: every query is compared against
every vector. That is O(n) per query and it is the right default here, because:

  - it is exact, so evaluation measures the retrieval strategy rather than an
    index's approximation error;
  - documentation and support corpora are tens of thousands of chunks, where
    exact search is milliseconds;
  - it has no dependencies, so `git clone && pytest` works on a fresh machine.

Approximate indexes (HNSW, IVF) earn their complexity somewhere north of a
million vectors. Reaching for one before that is buying recall loss with no
latency to show for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .chunking import Chunk
from .embed import Vector, cosine


@dataclass
class Hit:
    """One retrieved chunk with the score that put it there."""

    chunk: Chunk
    score: float
    source_scores: dict[str, float]  # per-retriever contributions, for debugging

    def __repr__(self) -> str:  # pragma: no cover - display only
        parts = ", ".join(f"{k}={v:.3f}" for k, v in self.source_scores.items())
        return f"Hit({self.chunk.id!r}, score={self.score:.4f}, {parts})"


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...
    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]: ...
    def get(self, index: int) -> Chunk: ...
    def __len__(self) -> int: ...


class InMemoryStore:
    """Exact cosine search over a Python list. See module docstring for why."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[Vector] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"got {len(chunks)} chunks and {len(vectors)} vectors; "
                f"they must correspond one to one"
            )
        if self._vectors and vectors and len(vectors[0]) != len(self._vectors[0]):
            raise ValueError(
                f"vector dimension {len(vectors[0])} does not match the "
                f"{len(self._vectors[0])} already in the store. Mixing embedding "
                f"models in one index silently destroys ranking; rebuild instead."
            )
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, vector: Vector, k: int) -> list[tuple[int, float]]:
        # A negative k would slice from the end and quietly return the *worst*
        # matches, which is the kind of bug that survives review because the
        # output still looks like results.
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        scored = ((i, cosine(vector, v)) for i, v in enumerate(self._vectors))
        return sorted(scored, key=lambda kv: -kv[1])[:k]

    def get(self, index: int) -> Chunk:
        return self._chunks[index]

    def __len__(self) -> int:
        return len(self._chunks)
