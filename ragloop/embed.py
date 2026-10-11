"""
Embeddings — the dense half, behind an interface small enough to fake.

An Embedder is any callable `(list[str]) -> list[list[float]]`. That is the whole
contract. It exists so that:

  - the retriever does not care whether vectors come from OpenAI, a local Ollama,
    or a hand-written stub;
  - the test suite runs with no network and no API key, which is the difference
    between tests that get run and tests that get skipped.

The default implementation talks to any OpenAI-compatible embeddings endpoint,
configured the same way as agentloop so both libraries read from one environment:

    RAGLOOP_BASE_URL     e.g. https://api.openai.com/v1  or  http://localhost:11434/v1
    RAGLOOP_API_KEY      key for that endpoint (local servers ignore it)
    RAGLOOP_EMBED_MODEL  e.g. text-embedding-3-small, nomic-embed-text
"""
from __future__ import annotations

import math
import os
from typing import Callable, Protocol, Sequence

Vector = list[float]


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> list[Vector]: ...


DEFAULT_EMBED_MODEL = os.environ.get("RAGLOOP_EMBED_MODEL", "text-embedding-3-small")


def openai_embedder(
    model: str | None = None, *, base_url: str | None = None, api_key: str | None = None,
    batch_size: int = 128,
) -> Embedder:
    """Embedder backed by an OpenAI-compatible endpoint.

    Batched because embedding one chunk per request is the most common reason a
    first indexing run takes twenty minutes instead of twenty seconds.
    """
    from openai import OpenAI  # imported lazily: importing ragloop must not require a key

    client = OpenAI(
        api_key=api_key or os.environ.get("RAGLOOP_API_KEY") or os.environ.get("OPENAI_API_KEY", "x"),
        base_url=base_url or os.environ.get("RAGLOOP_BASE_URL") or None,
    )
    model_id = model or DEFAULT_EMBED_MODEL

    def embed(texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = client.embeddings.create(model=model_id, input=batch)
            out.extend(ordered_embeddings(resp.data, len(batch)))
        return out

    return embed


def ordered_embeddings(data: Sequence, expected: int) -> list[Vector]:
    """Vectors in input order, from an embeddings response.

    OpenAI documents the order as preserved and also numbers each item with
    `index`; sorting by it makes the order explicit. Not every "OpenAI-compatible"
    endpoint fills `index` in, though: Gemini's leaves it None, and sorting on None
    raised TypeError on the first real batch. When any index is missing the
    response order is the only order there is, so it is used as is. A count
    mismatch is an error either way, because zipping a short list onto the chunks
    would attach vectors to the wrong text without a sound.
    """
    if len(data) != expected:
        raise ValueError(f"embeddings endpoint returned {len(data)} vectors for {expected} inputs")
    if all(getattr(d, "index", None) is not None for d in data):
        data = sorted(data, key=lambda d: d.index)
    return [d.embedding for d in data]


def hashing_embedder(dim: int = 256) -> Embedder:
    """A deterministic, dependency-free embedder for tests and offline demos.

    It is a bag-of-words hashing vectoriser, not a semantic model: it will match
    on shared vocabulary and will not match paraphrase. That is exactly what a
    test wants, because the assertion is about the retrieval machinery, not about
    a model's judgement. Never use it as a stand-in for a real embedder in an
    evaluation: it would flatter the lexical half of the pipeline.
    """
    from .bm25 import tokenize

    def embed(texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            vec = [0.0] * dim
            for token in tokenize(text):
                vec[hash_token(token) % dim] += 1.0
            out.append(normalise(vec))
        return out

    return embed


def hash_token(token: str) -> int:
    """FNV-1a. Python's builtin hash() is salted per process, which would make an
    index unusable across restarts."""
    h = 0x811C9DC5
    for ch in token.encode("utf-8"):
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return h


def normalise(vec: Vector) -> Vector:
    """Scale to unit length so cosine similarity reduces to a dot product."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity. Assumes nothing about normalisation, so it is safe to
    call on vectors straight from a provider."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)
