"""
Benchmark cross-encoder rerankers on top of EmbeddingGemma-300m dense retrieval.

For each query: dense top-K from the target snapshot, rerank those K with the
cross-encoder, and score (a) the reranked order and (b) an RRF fusion of dense and
reranker ranks. Latency is the reranking call alone (dense search is ~20 ms).

    .venv/Scripts/python dev/bench_rerankers.py                  # all rerankers
    .venv/Scripts/python dev/bench_rerankers.py minilm jina-v1   # subset (key prefixes)

Results: dev/results/rerankers.json (+ .md).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import retrieval_lib as R  # noqa: E402
import embedder as E  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
KS = (10, 20)
RERANKERS = {
    "minilm-l6": "Xenova/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "Xenova/ms-marco-MiniLM-L-12-v2",
    "jina-v1-tiny": "jinaai/jina-reranker-v1-tiny-en",
    "jina-v1-turbo": "jinaai/jina-reranker-v1-turbo-en",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "jina-v2-base": "jinaai/jina-reranker-v2-base-multilingual",
    "bge-v2-m3-int8": "BAAI/bge-reranker-v2-m3-int8",
}


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))]


def main(selected):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    chunks, dates = common.corpus()
    queries = R.load_queries()
    rels = [R.relevant(q, chunks, dates)[1] for q in queries]
    masks = [np.array([s in c.snapshots for c in chunks]) for s in range(len(dates))]
    emb = E.Embedder(log=lambda *_: None)
    D, _ = emb.embed_docs([E.doc_input(c.chunk) for c in chunks])
    dense = [R.rank(D @ emb.embed_query(q["q"]), masks[R.target_snapshot(q.get("as_of"), dates)])
             for q in queries]
    docs = [c.chunk.embed_text for c in chunks]

    OUT.mkdir(exist_ok=True)
    path = OUT / "rerankers.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results["dense only (EmbeddingGemma-300m)"] = {"metrics": R.metrics(dense, rels)}
    print("dense", results["dense only (EmbeddingGemma-300m)"]["metrics"], flush=True)

    for name, mid in RERANKERS.items():
        if selected and not any(name.startswith(s) for s in selected):
            continue
        print(f"== {name}", flush=True)
        try:
            t = time.time()
            ce = TextCrossEncoder(mid, cache_dir=str(common.CACHE / "models"))
            load_s = time.time() - t
            list(ce.rerank("warm up", docs[:4]))
            for K in KS:
                lat, rer, fused = [], [], []
                for qi, q in enumerate(queries):
                    top = dense[qi][:K]
                    t = time.time()
                    s = np.array(list(ce.rerank(q["q"], [docs[i] for i in top], batch_size=K)), np.float32)
                    lat.append(time.time() - t)
                    order = top[np.argsort(-s, kind="stable")]
                    rer.append(order)
                    fused.append(R.rrf(top, order, n=K))
                row = {"model": mid, "K": K, "load_s": round(load_s, 1),
                       "ms_median": round(1000 * statistics.median(lat)), "ms_p95": round(1000 * p95(lat)),
                       "rerank": R.metrics(rer, rels), "rrf(dense,rerank)": R.metrics(fused, rels)}
                results[f"{name} K={K}"] = row
                print(f"   K={K}: {row['ms_median']}ms (p95 {row['ms_p95']}) rerank {row['rerank']} "
                      f"rrf {row['rrf(dense,rerank)']}", flush=True)
        except Exception as e:
            results[name] = {"model": mid, "error": f"{type(e).__name__}: {e}"[:300]}
            print("   ERROR", results[name]["error"], flush=True)
        path.write_text(json.dumps(results, indent=2))
    write_md(results)


def write_md(results):
    d = results["dense only (EmbeddingGemma-300m)"]["metrics"]
    rows = [f"| dense only | — | — | {d['R@1']:.3f} / {d['R@5']:.3f} / {d['MRR@10']:.3f} | — |"]
    for k, r in results.items():
        if "rerank" not in r:
            if "error" in r:
                rows.append(f"| {k} | — | — | error: {r['error'][:80]} | |")
            continue
        a, b = r["rerank"], r["rrf(dense,rerank)"]
        rows.append(f"| {k} | {r['ms_median']} | {r['ms_p95']} | {a['R@1']:.3f} / {a['R@5']:.3f} / {a['MRR@10']:.3f} | "
                    f"{b['R@1']:.3f} / {b['R@5']:.3f} / {b['MRR@10']:.3f} |")
    md = ["# Reranker benchmark (on EmbeddingGemma-300m dense top-K)", "",
          "| reranker | ms median | ms p95 | rerank R@1 / R@5 / MRR | rrf(dense, rerank) R@1 / R@5 / MRR |",
          "|---|---|---|---|---|", *rows]
    (OUT / "rerankers.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
