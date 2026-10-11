from types import SimpleNamespace as NS

import pytest

from ragloop.embed import ordered_embeddings


def test_sorted_by_index_when_present():
    data = [NS(index=1, embedding=[1.0]), NS(index=0, embedding=[0.0])]
    assert ordered_embeddings(data, 2) == [[0.0], [1.0]]


def test_response_order_kept_when_index_missing():
    # Gemini's OpenAI-compatible endpoint returns index=None on every item.
    data = [NS(index=None, embedding=[0.0]), NS(index=None, embedding=[1.0])]
    assert ordered_embeddings(data, 2) == [[0.0], [1.0]]


def test_count_mismatch_is_an_error():
    with pytest.raises(ValueError):
        ordered_embeddings([NS(index=0, embedding=[0.0])], 2)
