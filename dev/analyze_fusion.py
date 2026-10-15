"""
Post-hoc analysis on cached document embeddings (run after bench_embedders.py):
  * weighted RRF (dense weight w vs BM25 weight 1) per model,
  * per-slice recall: public questions / re-worded variants / new questions / as-of questions.

    .venv/Scripts/python dev/analyze_fusion.py bge-small arctic-embed-m eg2-int8
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_embedders as B  # noqa: E402
import common  # noqa: E402
import retrieval_lib as R  # noqa: E402


def wrrf(dense, bm, w, k=60, n=100):
    acc = {}
    for weight, r in ((w, dense), (1.0, bm)):
        for pos, i in enumerate(r):
            acc[int(i)] = acc.get(int(i), 0.0) + weight / (k + pos + 1)
    return np.array(sorted(acc, key=lambda i: (-acc[i], i))[:n])


def slices(queries):
    out = {"all": list(range(len(queries)))}
    out["public"] = [i for i, q in enumerate(queries) if q["id"].startswith("p") and "." not in q["id"]]
    out["reworded"] = [i for i, q in enumerate(queries) if "." in q["id"]]
    out["new"] = [i for i, q in enumerate(queries) if q["id"].startswith("n") and "." not in q["id"]]
    out["as_of"] = [i for i, q in enumerate(queries) if q.get("as_of")]
    return out


def main(names):
    chunks, dates = common.corpus()
    B.DATES = dates
    queries = R.load_queries()
    rels = [R.relevant(q, chunks, dates)[1] for q in queries]
    masks = [np.array([s in c.snapshots for c in chunks]) for s in range(len(dates))]
    snap = [R.target_snapshot(q.get("as_of"), dates) for q in queries]
    bm = R.BM25([c.chunk.embed_text for c in chunks])
    bm_r = [R.rank(bm.scores(q["q"]), masks[snap[i]]) for i, q in enumerate(queries)]
    sig = hashlib.md5("".join(c.chunk.embed_text for c in chunks).encode()).hexdigest()[:10]
    sl = slices(queries)
    report = {}
    for name in names:
        kind, mid, qp, dp = B.MODELS[name]
        all_texts = B.doc_texts(chunks, dp)
        texts = sorted(set(all_texts))
        pos = {t: i for i, t in enumerate(texts)}
        d = B.normalize(np.load(B.EMB_DIR / f"{name}-{sig}.npy")[[pos[t] for t in all_texts]])
        model = B.EG2Onnx(mid) if kind == "eg2" else B.FastEmbedder(mid)
        qv = B.normalize(np.array([model.embed([qp + q["q"]])[0] for q in queries]))
        dense = [R.rank(d @ qv[i], masks[snap[i]]) for i in range(len(queries))]
        rows = {"bm25": bm_r, "dense": dense}
        for w in (1.0, 1.5, 2.0, 3.0):
            rows[f"rrf w={w}"] = [wrrf(dense[i], bm_r[i], w) for i in range(len(queries))]
        report[name] = {}
        print(f"\n== {name}")
        print(f"{'method':12}" + "".join(f"{s:>22}" for s in sl))
        for meth, rk in rows.items():
            cells = []
            for s, idx in sl.items():
                m = R.metrics([rk[i] for i in idx], [rels[i] for i in idx])
                cells.append(f"{m['R@5']:.2f}/{m['R@10']:.2f}/{m['MRR@10']:.2f}")
                report[name].setdefault(meth, {})[s] = m
            print(f"{meth:12}" + "".join(f"{c:>22}" for c in cells))
        misses = [queries[i]["id"] for i in range(len(queries))
                  if not any(int(j) in rels[i] for j in rows["rrf w=2.0"][i][:10])]
        print("misses@10 (rrf w=2):", misses)
        report[name]["misses@10"] = misses
    (B.OUT / "fusion.json").write_text(json.dumps(report, indent=2))
    print("\ncells = R@5 / R@10 / MRR@10;  slices sizes:", {k: len(v) for k, v in sl.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
