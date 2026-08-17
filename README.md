# ragloop

Hybrid retrieval with the measurement attached. BM25 and dense vectors fused by
reciprocal rank, plus the evaluation harness that tells you whether any of it
helped. No dependencies: the whole index is a few hundred lines of readable
Python, and `git clone && pytest` works on a fresh machine.

Companion to [agentloop](https://github.com/Efecem07/agentloop), same idea
applied one layer down: the interesting part of an AI system is not the model
call, it is everything around it, and none of it should be argued about when it
can be measured.

## The claim this library is built to test

"We added RAG" is not checkable. This is:

```
=== RETRIEVAL ABLATION ===
  lexical   recall@5=0.750  MRR=0.719  nDCG@5=0.727  (16 queries, 4 with no hit)
  dense     recall@5=0.844  MRR=0.690  nDCG@5=0.721  (16 queries, 2 with no hit)
  hybrid    recall@5=0.844  MRR=0.762  nDCG@5=0.775  (16 queries, 2 with no hit)
  hybrid vs lexical: recall +0.094  ->  worth it

  still missed by hybrid (2):
    how long until a refund shows up
    when do I get my money back
```

Reproduce with `python -m examples.demo`. Two things in that output matter more
than the averages: the ablation says whether the second index is paying for
itself, and the miss list names the queries that are still broken. An average
tells you how you are doing; the miss list tells you what to do next.

**These numbers are a floor, not a result.** With no embeddings key the demo
substitutes a hashing vectoriser for the dense half, which shares vocabulary
with BM25 and is not a semantic model. Set `RAGLOOP_BASE_URL` and
`RAGLOOP_API_KEY` (or `OPENAI_API_KEY`) and the dense and hybrid rows become
meaningful. The harness is the point; the corpus here is a fixture.

## Why hybrid

The two retrievers fail in opposite directions, which is the entire argument for
paying for both.

| | fails on | example |
|---|---|---|
| dense | literals | `TS2345`, `§ 5a`, an order number, a rare product name. The embedding of a rare token carries almost no signal, so the neighbours are merely on-topic. |
| lexical | paraphrase | "my card was declined" against a document that says "the issuing bank rejects the transaction". Zero shared terms. |

### Fused by rank, not by score

BM25 returns unbounded scores whose scale depends on corpus statistics. Cosine
returns roughly 0 to 1. Adding them means normalising two distributions that
shift every time documents are added, and the weight tuned on Tuesday is wrong on
Friday. Reciprocal rank fusion discards the scores and keeps the ranks:

```
score(d) = Σ over retrievers of  1 / (k + rank(d))        k = 60
```

One parameter, immune to scale changes, and a document both retrievers rank
moderately well beats a document only one of them loves. That last property is
the one that shows up in the numbers.

## Quickstart

```python
from ragloop import chunk_documents, HybridIndex, openai_embedder, build_context

chunks = chunk_documents({"handbook.md": text}, size=400, overlap=60)

index = HybridIndex(embedder=openai_embedder())
index.add(chunks)

hits = index.search("how do I rotate an API token", k=5)
for hit in hits:
    print(hit)          # Hit('security.md#0', score=0.0325, bm25=0.016, dense=0.016)

context = build_context(hits, max_chars=4000)   # citable [source#n] blocks
```

Every hit carries `source_scores`, the per-retriever contribution that produced
its rank. When a result is wrong, that field says which half went wrong.

```bash
pip install -e ".[dev,openai]"
pytest

export RAGLOOP_BASE_URL=https://api.openai.com/v1
export RAGLOOP_API_KEY=sk-...
export RAGLOOP_EMBED_MODEL=text-embedding-3-small

python -m examples.demo --markdown
python -m examples.demo --show "my card was declined"
```

Any OpenAI-compatible embeddings endpoint works, including a local Ollama at
`http://localhost:11434/v1` with `nomic-embed-text`.

## Measuring it

```python
from ragloop import QueryCase, ablation, print_ablation

cases = [
    QueryCase("TS2345", relevant={"errors.md#0"}),
    QueryCase("my card was declined", relevant={"billing.md#0"}),
]
print_ablation(ablation(index, cases, k=5))
```

`QueryCase.relevant` holds chunk ids. Building that set by reading the documents
is the actual work of evaluating retrieval, and there is no shortcut: thirty
labelled queries tell you more than any amount of looking at outputs.

| metric | question it answers |
|---|---|
| recall@k | did the answer reach the context window at all? This is the ceiling on everything downstream. A generator cannot cite what it never saw, so a retrieval bug looks exactly like a hallucination and gets misdiagnosed as one. |
| MRR | how high did the first relevant chunk land? Models attend unevenly across a long context; rank 1 is not rank 9. |
| nDCG@k | the graded version: several relevant chunks near the top, discounted by position. |

Recall here means "fraction of relevant chunks retrieved", not "at least one hit".
The lenient reading scores a half-answered two-document question as a perfect
1.0, which is exactly the failure worth catching.

## Design notes

**Chunking decides the ceiling.** If the sentence that answers the question is
split across two chunks, no reranker and no better embedding model recovers it.
So chunks are packed on sentence boundaries, paragraph breaks are hard
boundaries, and a sentence longer than the budget is kept whole rather than cut.
Overlap trades index size for recall, and recall is the thing you cannot fix
later in the pipeline.

**BM25 is implemented, not imported.** Sixty lines, an inverted index, and the
three ideas it rests on are in the module docstring: IDF, saturation, length
normalisation. The `+1` smoothing in the IDF is there so a term appearing in
most documents cannot score negative and push a matching document below a
non-matching one, which reads as a bug to everyone who hits it.

**Exact search, on purpose.** `InMemoryStore` compares the query against every
vector. That is O(n) and it is right for the documentation and support corpora
this targets: it is exact, so the evaluation measures the retrieval strategy
rather than an index's approximation error. HNSW and IVF earn their complexity
somewhere north of a million vectors; reaching for one earlier buys recall loss
with no latency to show for it. `VectorStore` is four methods wide, so a managed
database drops in as an adapter.

**The embedder is injectable.** Any `(list[str]) -> list[list[float]]` callable.
That is why the 43 tests run with no network and no API key, which is the
difference between tests that get run and tests that get skipped.

**Fusion depth is separate from k.** Each retriever is asked for 30 candidates
before fusion, not for the final 5. Fusing only the top 5 from each would discard
exactly the documents RRF exists to rescue: the ones ranked 8th by both rather
than 1st by one.

## Limitations

- **Not run in production.** This is a working reference pipeline with a test
  suite and an eval harness, not a system that has carried live traffic. Said
  plainly rather than left to be inferred.
- **No reranker.** A cross-encoder over the fused top 30 is the obvious next
  step and usually the largest single gain after hybrid itself.
- **No generation.** `build_context` hands you a citable context block; what you
  do with it is out of scope, and grounding metrics for the generated answer are
  a different harness.
- **In-memory only.** No persistence, so an index is rebuilt on start.
- **German and English tokenisation.** The stopword list is small and bilingual;
  there is no stemming, deliberately, because stemming helps prose recall and
  hurts identifier matching.

## License

MIT.
