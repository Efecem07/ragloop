"""Run the ablation: lexical, dense and hybrid over the same labelled queries.

    python -m examples.demo                 # offline, hashing stub for the dense half
    python -m examples.demo --markdown      # also print a Markdown table
    python -m examples.demo --k 3

With RAGLOOP_BASE_URL / RAGLOOP_API_KEY (or OPENAI_API_KEY) set, the dense half
uses real embeddings and the ablation means something. Without a key it falls
back to a hashing stub and says so loudly, because a hashing vectoriser shares
vocabulary with BM25 and would flatter the dense row into meaninglessness.
"""
from __future__ import annotations

import argparse
import os
import sys

from ragloop import (
    HybridIndex,
    QueryCase,
    ablation,
    chunk_documents,
    hashing_embedder,
    markdown_ablation,
    openai_embedder,
    print_ablation,
)

from .corpus import DOCUMENTS, QUERIES


def pick_embedder() -> tuple[object, bool]:
    """Return (embedder, is_real). Never guess: a stub must announce itself."""
    if os.environ.get("RAGLOOP_BASE_URL") or os.environ.get("OPENAI_API_KEY") or os.environ.get("RAGLOOP_API_KEY"):
        try:
            return openai_embedder(), True
        except Exception as e:  # noqa: BLE001 - missing client or bad config
            print(f"! real embedder unavailable ({e}); falling back to the stub", file=sys.stderr)
    return hashing_embedder(), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--show", metavar="QUERY", help="print the top hits for one query and exit")
    args = ap.parse_args()

    chunks = chunk_documents(DOCUMENTS, size=400, overlap=60)
    ids = {c.id for c in chunks}

    # The labels in corpus.py name chunk ids, which depend on the chunking
    # parameters. If those drift, the eval silently measures nothing, so check.
    labelled = {cid for _q, rel, _kind in QUERIES for cid in rel}
    missing = labelled - ids
    if missing:
        print(f"! labels reference chunk ids that do not exist: {sorted(missing)}", file=sys.stderr)
        print(f"  chunking produced: {sorted(ids)}", file=sys.stderr)
        return 1

    embedder, is_real = pick_embedder()
    index = HybridIndex(embedder=embedder)
    index.add(chunks)

    if args.show:
        for hit in index.search(args.show, k=args.k):
            print(f"  {hit}\n    {hit.chunk.text[:140]}...")
        return 0

    cases = [QueryCase(q, rel, note=kind) for q, rel, kind in QUERIES]
    results = ablation(index, cases, k=args.k)

    print(f"\ncorpus: {len(DOCUMENTS)} documents -> {len(chunks)} chunks, {len(cases)} labelled queries")
    if not is_real:
        print("dense half: HASHING STUB, not a semantic model. The dense and hybrid")
        print("            rows below are lower bounds; set an embeddings key for real numbers.")
    print_ablation(results)

    # Where hybrid actually earns its keep: queries no single retriever answers.
    lex_misses = set(results["lexical"].misses)
    hyb_misses = set(results["hybrid"].misses)
    rescued = lex_misses - hyb_misses
    if rescued:
        print(f"\n  rescued by the dense half ({len(rescued)}):")
        for q in sorted(rescued):
            print(f"    {q}")
    if hyb_misses:
        print(f"\n  still missed by hybrid ({len(hyb_misses)}):")
        for q in sorted(hyb_misses):
            print(f"    {q}")

    if args.markdown:
        print("\n" + markdown_ablation(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
