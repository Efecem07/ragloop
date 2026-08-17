"""Chunking decides the ceiling on retrieval, so its edges are tested explicitly."""
from __future__ import annotations

import pytest

from ragloop import Chunk, chunk_documents, chunk_text, split_sentences

TEXT = (
    "Password resets are handled in the admin panel. "
    "Open Settings, then Security, then Reset. "
    "The user receives a mail within five minutes.\n\n"
    "Billing questions go to a different team. "
    "Invoices are issued on the first of each month."
)


def test_sentences_split_on_terminators_and_paragraphs():
    sents = split_sentences(TEXT)
    assert sents[0] == "Password resets are handled in the admin panel."
    assert "Billing questions go to a different team." in sents
    assert len(sents) == 5


def test_paragraph_break_is_a_hard_boundary():
    # A heading immediately followed by prose must not be glued to the previous
    # paragraph's last sentence.
    sents = split_sentences("First para ends here\n\nSecond para starts here")
    assert sents == ["First para ends here", "Second para starts here"]


def test_chunks_respect_the_size_budget():
    chunks = chunk_text(TEXT, size=120, overlap=20, source="faq.md")
    assert len(chunks) > 1
    # Overlap is carried in, so a chunk may exceed size by at most the overlap.
    assert all(len(c) <= 120 + 20 for c in chunks)


def test_overlap_repeats_boundary_context():
    chunks = chunk_text(TEXT, size=120, overlap=40, source="faq.md")
    tail = chunks[0].text[-20:]
    assert tail.split()[-1] in chunks[1].text


def test_no_overlap_means_no_repetition():
    chunks = chunk_text(TEXT, size=120, overlap=0)
    joined = " ".join(c.text for c in chunks)
    assert joined.count("Password resets are handled") == 1


def test_oversized_sentence_is_kept_whole_not_cut():
    long_sentence = "word " * 300 + "end."
    chunks = chunk_text(long_sentence, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("end.")


def test_chunk_ids_are_stable_and_citable():
    chunks = chunk_text(TEXT, size=120, overlap=20, source="faq.md")
    assert chunks[0].id == "faq.md#0"
    assert chunks[1].id == "faq.md#1"
    assert Chunk("x").id == "0"


def test_chunk_documents_flattens_and_keeps_provenance():
    chunks = chunk_documents({"a.md": TEXT, "b.md": "Short doc."}, size=200, overlap=20)
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.md"}


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (100, -1), (100, 100), (100, 150)],
)
def test_impossible_parameters_are_rejected_at_the_call(size, overlap):
    # overlap >= size would mean every chunk re-emits its predecessor and the
    # loop never advances. Better to refuse than to hang.
    with pytest.raises(ValueError):
        chunk_text(TEXT, size=size, overlap=overlap)


def test_empty_input_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []
