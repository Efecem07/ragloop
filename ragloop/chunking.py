"""
Chunking — turning documents into the units a retriever actually scores.

This is the least glamorous part of RAG and the one that decides most of the
outcome. Retrieval quality is bounded by chunking: if the sentence that answers
the question is split across two chunks, no reranker and no better embedding
model will recover it.

Two choices are made here, both deliberate:

  SENTENCE BOUNDARIES, NOT CHARACTER COUNTS. Cutting mid-sentence produces
  chunks that embed badly (half a thought has no clear topic) and read badly
  when shown to a user as a citation. So text is split into sentences first and
  packed into chunks up to the size budget.

  OVERLAP. A fact that sits at a chunk boundary is otherwise invisible to both
  neighbours. Overlap trades index size for recall, and recall is what you
  cannot fix later in the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sentence terminator followed by whitespace and a capital/quote/digit. Not a
# linguistic parser: a deliberately small rule that fails predictably rather
# than a dependency that fails surprisingly.
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-ZÄÖÜ0-9])")
_WS = re.compile(r"[ \t]+")


@dataclass
class Chunk:
    """One retrievable unit, with enough provenance to cite it."""

    text: str
    source: str = ""
    index: int = 0                          # position within its source document
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}" if self.source else str(self.index)

    def __len__(self) -> int:
        return len(self.text)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping paragraph breaks as hard boundaries.

    A paragraph break is a stronger signal than a full stop: a heading followed
    by a paragraph should never be packed into the middle of an unrelated chunk.
    """
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = _WS.sub(" ", para).strip()
        if not para:
            continue
        for sent in _SENT_END.split(para):
            sent = sent.strip()
            if sent:
                out.append(sent)
    return out


def chunk_text(
    text: str,
    *,
    source: str = "",
    size: int = 400,
    overlap: int = 60,
    meta: dict | None = None,
) -> list[Chunk]:
    """Pack sentences into chunks of at most `size` characters, with `overlap`
    characters of trailing context repeated at the start of the next chunk.

    A single sentence longer than `size` is kept whole rather than cut: an
    over-long chunk is a retrievable chunk, half a sentence is not.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size, otherwise chunks never advance")

    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(buf).strip()
        if body:
            chunks.append(Chunk(body, source=source, index=len(chunks), meta=dict(meta or {})))
        # Carry the tail of this chunk into the next one so a fact sitting on the
        # boundary is reachable from both sides.
        if overlap and body:
            tail = body[-overlap:]
            cut = tail.find(" ")
            tail = tail[cut + 1:] if cut != -1 else tail
            buf = [tail] if tail else []
            buf_len = len(tail)
        else:
            buf, buf_len = [], 0

    for sent in sentences:
        if buf and buf_len + 1 + len(sent) > size:
            flush()
        buf.append(sent)
        buf_len += len(sent) + (1 if buf_len else 0)

    # Final flush must not re-seed the buffer, hence the explicit tail handling.
    if buf:
        body = " ".join(buf).strip()
        if body:
            chunks.append(Chunk(body, source=source, index=len(chunks), meta=dict(meta or {})))

    return chunks


def chunk_documents(
    docs: dict[str, str], *, size: int = 400, overlap: int = 60
) -> list[Chunk]:
    """Chunk a {source_name: text} mapping into one flat list."""
    out: list[Chunk] = []
    for source, text in docs.items():
        out.extend(chunk_text(text, source=source, size=size, overlap=overlap))
    return out
