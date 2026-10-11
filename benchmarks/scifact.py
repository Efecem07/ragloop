"""Lexical vs dense vs hybrid on BEIR SciFact, a public benchmark.

The demo corpus in examples/ is a fixture: sixteen hand-written queries prove the
harness works, not that hybrid retrieval does. This runs the same `ablation()` on a
dataset other people report numbers for, so the result can be checked against them.

SciFact: 5,183 scientific abstracts, 300 test claims, each labelled with the
abstracts that support or refute it (Wadden et al., 2020; BEIR, Thakur et al., 2021).

    python -m benchmarks.scifact                       # BM25 only, no key needed
    RAGLOOP_API_KEY=... RAGLOOP_BASE_URL=... \\
    RAGLOOP_EMBED_MODEL=... python -m benchmarks.scifact --markdown

Any OpenAI-compatible embeddings endpoint works. Vectors are cached on disk under
benchmarks/.cache keyed by model and text, so a second run costs nothing and the
numbers can be regenerated exactly. On a rate-limited free tier set RAGLOOP_BATCH
(texts per request) and RAGLOOP_TPM (tokens per minute) and it paces itself.

One abstract is one chunk (title + text), so a hit is a document and the labels
apply as published. Chunking would change what is being measured.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from ragloop import Chunk, HybridIndex, QueryCase, ablation, markdown_ablation, openai_embedder, print_ablation

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
CACHE = Path(__file__).with_name(".cache")


def load_scifact() -> tuple[list[Chunk], list[QueryCase]]:
    """Download once, then read corpus, test queries and qrels from the zip."""
    CACHE.mkdir(exist_ok=True)
    zpath = CACHE / "scifact.zip"
    if not zpath.exists():
        print(f"downloading {URL}", file=sys.stderr)
        with urllib.request.urlopen(URL, timeout=120) as r:
            zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath) as z:
        def lines(name: str) -> list[str]:
            return z.read(f"scifact/{name}").decode("utf-8").splitlines()

        chunks = []
        for line in lines("corpus.jsonl"):
            d = json.loads(line)
            text = f"{d.get('title', '')}. {d.get('text', '')}".strip(". ")
            chunks.append(Chunk(text=text, source=str(d["_id"]), index=0))
        queries = {str(q["_id"]): q["text"] for q in map(json.loads, lines("queries.jsonl"))}
        relevant: dict[str, set[str]] = {}
        for row in lines("qrels/test.tsv")[1:]:  # header: query-id corpus-id score
            qid, did, score = row.split("\t")
            if int(score) > 0:
                relevant.setdefault(qid, set()).add(f"{did}#0")
    cases = [QueryCase(queries[q], rel) for q, rel in sorted(relevant.items()) if q in queries]
    return chunks, cases


def cached(embed, model: str):
    """Wrap an embedder with a gzip JSON cache and patient retries.

    Free tiers rate-limit hard; a benchmark that dies at batch 40 of 52 and starts
    over is how a ten-minute job takes an afternoon.
    """
    path = CACHE / f"vectors-{hashlib.sha1(model.encode()).hexdigest()[:10]}.json.gz"
    store: dict[str, list[float]] = {}
    if path.exists():
        store = json.loads(gzip.decompress(path.read_bytes()))

    def key(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    def save() -> None:
        path.write_bytes(gzip.compress(json.dumps(store).encode()))

    # Tokens per minute the endpoint allows (RAGLOOP_TPM, unset = no pacing). Estimated
    # as characters / 4, which is close enough for English abstracts.
    tpm = int(os.environ.get("RAGLOOP_TPM", "0"))
    size = int(os.environ.get("RAGLOOP_BATCH", "100"))
    window: list[tuple[float, int]] = []

    def pace(tokens: int) -> None:
        if not tpm:
            return
        while True:
            now = time.time()
            window[:] = [(t, n) for t, n in window if now - t < 60]
            if sum(n for _, n in window) + tokens <= tpm:
                window.append((now, tokens))
                return
            time.sleep(1)

    def wrapped(texts):
        todo = [t for t in dict.fromkeys(texts) if key(t) not in store]
        for i in range(0, len(todo), size):
            batch = todo[i : i + size]
            pace(sum(len(t) for t in batch) // 4 + 1)
            for attempt in range(20):
                try:
                    for t, v in zip(batch, embed(batch)):
                        store[key(t)] = v
                    break
                except Exception as e:  # noqa: BLE001 - rate limits arrive as many types
                    wait = min(60, 5 * 2 ** attempt)
                    print(f"  embed retry in {wait}s: {str(e)[:80]}", file=sys.stderr)
                    time.sleep(wait)
            else:
                save()
                raise RuntimeError("embedding kept failing; progress is cached, run again")
            if len(todo) > size and (i // size) % 10 == 0:
                print(f"  embedded {min(i + size, len(todo))}/{len(todo)}", file=sys.stderr)
                save()
        if todo:
            save()
        return [store[key(t)] for t in texts]

    return wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    chunks, cases = load_scifact()
    has_key = bool(os.environ.get("RAGLOOP_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("RAGLOOP_EMBED_MODEL", "text-embedding-3-small")
    # Retrieval models trained asymmetrically (NVIDIA's nemotron/embedqa, E5) want to be
    # told whether a text is a document or a question; RAGLOOP_INPUT_TYPE=1 sends
    # input_type=passage while indexing and input_type=query while searching. The two
    # modes are cached separately, since the same text gets a different vector.
    mode = {"input_type": "passage"}
    embedder = None
    if has_key and os.environ.get("RAGLOOP_INPUT_TYPE") == "1":
        from openai import OpenAI
        from ragloop.embed import ordered_embeddings

        client = OpenAI(api_key=os.environ.get("RAGLOOP_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("RAGLOOP_BASE_URL") or None)

        def raw(texts):
            resp = client.embeddings.create(model=model, input=list(texts),
                                            extra_body={**mode, "truncate": "END"})
            return ordered_embeddings(resp.data, len(texts))

        per_mode = {m: cached(raw, f"{model}:{m}") for m in ("passage", "query")}
        embedder = lambda texts: per_mode[mode["input_type"]](texts)  # noqa: E731
    elif has_key:
        embedder = cached(openai_embedder(model, batch_size=10_000), model)
    print(f"SciFact: {len(chunks)} abstracts, {len(cases)} test queries, "
          f"dense={'%s' % model if embedder else 'off (no key)'}", file=sys.stderr)

    index = HybridIndex(embedder=embedder)
    index.add(chunks)
    mode["input_type"] = "query"
    results = ablation(index, cases, k=args.k)
    print_ablation(results, title=f"SCIFACT (BEIR test, k={args.k})")
    if args.markdown:
        print()
        print(markdown_ablation(results, title=f"BEIR SciFact, {len(cases)} test queries, embeddings: {model if embedder else 'none'}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
