"""ragloop — hybrid retrieval with a measurement harness attached.

BM25 and dense vectors fused by reciprocal rank, plus the evaluation that tells
you whether any of it helped. No dependencies beyond an embeddings client.
"""
from .bm25 import BM25, tokenize
from .chunking import Chunk, chunk_documents, chunk_text, split_sentences
from .embed import Embedder, cosine, hashing_embedder, openai_embedder
from .evals import (
    Metrics,
    QueryCase,
    ablation,
    evaluate,
    markdown_ablation,
    print_ablation,
)
from .retrieve import HybridIndex, build_context
from .store import Hit, InMemoryStore, VectorStore

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "chunk_text",
    "chunk_documents",
    "split_sentences",
    "BM25",
    "tokenize",
    "Embedder",
    "openai_embedder",
    "hashing_embedder",
    "cosine",
    "HybridIndex",
    "build_context",
    "Hit",
    "InMemoryStore",
    "VectorStore",
    "QueryCase",
    "Metrics",
    "evaluate",
    "ablation",
    "print_ablation",
    "markdown_ablation",
    "__version__",
]
