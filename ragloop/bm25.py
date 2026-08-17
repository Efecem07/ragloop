"""
BM25 — the lexical half of hybrid retrieval, implemented rather than imported.

Embeddings are good at meaning and bad at literals. Ask for "error TS2345" or
"paragraph 5a" or a product code, and a dense retriever will happily return
something that is *about* type errors or *about* paragraphs. BM25 will return
the document that contains the string. Most real corpora need both, which is
why this file exists next to embed.py.

BM25 scores a document for a query term as:

    IDF(t) * ( f(t,d) * (k1 + 1) ) / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

Three ideas, and they are worth stating because they explain the tuning knobs:

  IDF          a term that appears in every document tells you nothing; a rare
               term is nearly the whole signal.
  SATURATION   the tenth occurrence of a word means much less than the second.
               k1 sets how fast the payoff flattens (higher k1 = slower).
  LENGTH NORM  a long document contains more words by accident, so raw counts
               favour it unfairly. b sets how hard we correct for that
               (b=0 no correction, b=1 full correction).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9äöüß]+")

# Deliberately tiny and bilingual: the corpora this is aimed at are product and
# support documents, where a big stopword list removes terms that turn out to
# matter ("not", "no", "kein"). Keeping the list short is the safer failure.
STOPWORDS = frozenset("""
a an and are as at be by for from has have in is it its of on or that the to was were with
der die das und ist sind im in den dem des ein eine einer eines auf mit von zu fuer für
""".split())


def tokenize(text: str, *, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords. No stemming on purpose:
    stemming helps recall on prose and hurts on identifiers, and identifiers are
    exactly what the lexical half is here to catch."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in stopwords]


class BM25:
    """BM25-Okapi over an in-memory corpus.

    Built for corpora that fit in memory, which covers the documentation and
    support-article case this library targets. The inverted index keeps this
    linear in the number of documents containing a query term, not in corpus
    size, so a query stays fast as the long tail of documents grows.
    """

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens: list[list[str]] = [tokenize(d) for d in documents]
        self.doc_len: list[int] = [len(t) for t in self.docs_tokens]
        self.n = len(documents)
        self.avgdl: float = (sum(self.doc_len) / self.n) if self.n else 0.0

        # term -> {doc_index: term frequency}
        self.postings: dict[str, dict[int, int]] = {}
        for i, tokens in enumerate(self.docs_tokens):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, {})[i] = freq

        # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative for
        # terms appearing in more than half the corpus. Without it, a very common
        # term scores negative and can push a matching document below a
        # non-matching one, which reads as a bug to everyone who hits it.
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self.postings.items()
        }

    def scores(self, query: str) -> dict[int, float]:
        """Return {doc_index: score} for documents sharing at least one term."""
        out: dict[int, float] = {}
        if not self.n:
            return out
        # Guard the divisor once, explicitly. avgdl is only zero when every
        # document tokenises to nothing, in which case there are no postings to
        # score anyway, but an expression that reads as a guard should be one.
        avgdl = self.avgdl if self.avgdl > 0 else 1.0
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_i, freq in postings.items():
                norm = 1 - self.b + self.b * (self.doc_len[doc_i] / avgdl)
                out[doc_i] = out.get(doc_i, 0.0) + idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * norm
                )
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Top-k (doc_index, score), best first."""
        ranked = sorted(self.scores(query).items(), key=lambda kv: -kv[1])
        return ranked[:k]
