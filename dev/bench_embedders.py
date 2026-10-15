"""
Benchmark local embedding models on the snapshot corpus.

For each model: embed every unique chunk (this is the ingest cost), embed each dev
query on its own (answer-time cost), then rank only the chunks live in the query's
target snapshot. Quality is reported dense-only and fused with BM25 (RRF), since the
solution will use the hybrid.

    .venv/Scripts/python dev/bench_embedders.py                 # all models
    .venv/Scripts/python dev/bench_embedders.py bge-small eg2   # a subset (name prefixes)

Results: dev/results/embedders.json (+ .md). Embeddings are cached in dev/.cache/emb/.
"""
from __future__ import annotations

import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import retrieval_lib as R  # noqa: E402

MODEL_DIR = common.CACHE / "models"
EMB_DIR = common.CACHE / "emb"
OUT = Path(__file__).resolve().parent / "results"
MAX_INGEST_S = 25 * 60       # skip models projected to blow the 30-minute ingest limit
BGE_Q = "Represent this sentence for searching relevant passages: "
GEMMA_Q = "task: search result | query: "
GEMMA_D = "title: {title} | text: "

# name -> (kind, model id / file, query prefix, doc prefix)
MODELS = {
    "potion-retrieval-32M": ("fe", "minishlab/potion-retrieval-32M", "", ""),
    "all-MiniLM-L6-v2": ("fe", "sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "arctic-embed-xs": ("fe", "snowflake/snowflake-arctic-embed-xs", BGE_Q, ""),
    "bge-small-en-v1.5": ("fe", "BAAI/bge-small-en-v1.5", BGE_Q, ""),
    "arctic-embed-s": ("fe", "snowflake/snowflake-arctic-embed-s", BGE_Q, ""),
    "granite-small-r2": ("fe", "ibm-granite/granite-embedding-small-english-r2", "", ""),
    "jina-v2-small": ("fe", "jinaai/jina-embeddings-v2-small-en", "", ""),
    "bge-base-en-v1.5": ("fe", "BAAI/bge-base-en-v1.5", BGE_Q, ""),
    "arctic-embed-m": ("fe", "snowflake/snowflake-arctic-embed-m", BGE_Q, ""),
    "gte-base": ("fe", "thenlper/gte-base", "", ""),
    "nomic-v1.5-Q": ("fe", "nomic-ai/nomic-embed-text-v1.5-Q", "search_query: ", "search_document: "),
    "embeddinggemma-300m": ("fe", "google/embeddinggemma-300m", GEMMA_Q, GEMMA_D),
    "eg2-int8": ("eg2", "model_quantized.onnx", GEMMA_Q, GEMMA_D),
    "eg2-fp32": ("eg2", "model.onnx", GEMMA_Q, GEMMA_D),
    "mxbai-embed-large": ("fe", "mixedbread-ai/mxbai-embed-large-v1", BGE_Q, ""),
    "qwen3-0.6B-Q": ("fe", "Qwen/Qwen3-Embedding-0.6B-Q",
                     "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", ""),
}
# alternate query prompts scored on the same document vectors (cheap: queries only)
QUERY_VARIANTS = {"eg2-fp32": {"qa-prompt": "task: question answering | query: "},
                  "eg2-int8": {"qa-prompt": "task: question answering | query: "}}


class FastEmbedder:
    def __init__(self, model_id):
        from fastembed import TextEmbedding
        self.m = TextEmbedding(model_id, cache_dir=str(MODEL_DIR))

    def embed(self, texts, batch_size=8):
        # length-sorted small batches: ~2x faster than fastembed's default (less padding)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = np.array(list(self.m.embed([texts[i] for i in order], batch_size=batch_size)), dtype=np.float32)
        out = np.empty_like(vecs)
        out[order] = vecs
        return out


class EG2Onnx:
    """EmbeddingGemma 2 text model via onnxruntime (fastembed cannot feed its extra inputs)."""
    REPO = "onnx-community/embeddinggemma-2-ONNX"

    def __init__(self, filename, max_tokens=1024):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        d = Path(snapshot_download(self.REPO, local_dir=str(MODEL_DIR / "embeddinggemma-2-ONNX"),
                                   allow_patterns=[f"onnx/{filename}*", "tokenizer.json"]))
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_tokens)
        self.tok.enable_padding(pad_id=self.tok.token_to_id("<pad>") or 0, pad_token="<pad>")
        self.s = ort.InferenceSession(str(d / "onnx" / filename), providers=["CPUExecutionProvider"])
        self.extra = {i.name: np.zeros((0, 512), np.float32) for i in self.s.get_inputs()[2:]}

    def embed(self, texts, batch_size=8):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-bucketed batches
        out = np.zeros((len(texts), 768), np.float32)
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            enc = self.tok.encode_batch([texts[i] for i in idx])
            feed = {"input_ids": np.array([e.ids for e in enc], np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], np.int64), **self.extra}
            out[idx] = self.s.run(["sentence_embedding"], feed)[0]
        return out


def doc_texts(chunks, doc_prefix):
    if "{title}" in doc_prefix:
        return [doc_prefix.format(title=c.chunk.title or "none")
                + (f"{c.chunk.section}\n" if c.chunk.section else "") + R_strip(c.chunk.text) for c in chunks]
    return [doc_prefix + c.chunk.embed_text for c in chunks]


def R_strip(text):
    from corpus import strip_links
    return strip_links(text)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def evaluate(qvecs, dvecs, queries, rels, masks, bm25_ranks):
    dense, hybrid = [], []
    for qi, q in enumerate(queries):
        mask = masks[R.target_snapshot(q.get("as_of"), DATES)]
        d = R.rank(dvecs @ qvecs[qi], mask)
        dense.append(d)
        hybrid.append(R.rrf(d, bm25_ranks[qi]))
    return R.metrics(dense, rels), R.metrics(hybrid, rels)


def main(selected):
    global DATES
    chunks, DATES = common.corpus()
    queries = R.load_queries()
    rels = [R.relevant(q, chunks, DATES)[1] for q in queries]
    masks = [np.array([s in c.snapshots for c in chunks]) for s in range(len(DATES))]
    sig = hashlib.md5("".join(c.chunk.embed_text for c in chunks).encode()).hexdigest()[:10]
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    res_path = OUT / "embedders.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    t = time.time()
    bm = R.BM25([c.chunk.embed_text for c in chunks])
    bm_build = time.time() - t
    bm_ranks, bm_lat = [], []
    for q in queries:
        t = time.time()
        bm_ranks.append(R.rank(bm.scores(q["q"]), masks[R.target_snapshot(q.get("as_of"), DATES)]))
        bm_lat.append(time.time() - t)
    results["bm25"] = {"model": "BM25 (own, k1=1.2 b=0.75)", "ingest_s": round(bm_build, 1), "dim": 0,
                       "query_ms": round(1000 * statistics.median(bm_lat), 1),
                       "dense": R.metrics(bm_ranks, rels), "hybrid": R.metrics(bm_ranks, rels)}
    print(f"bm25: {results['bm25']['dense']}", flush=True)

    for name, (kind, mid, qp, dp) in MODELS.items():
        if selected and not any(name.startswith(s) for s in selected):
            continue
        print(f"== {name}", flush=True)
        try:
            t = time.time()
            model = EG2Onnx(mid) if kind == "eg2" else FastEmbedder(mid)
            load_s = time.time() - t
            all_texts = doc_texts(chunks, dp)
            texts = sorted(set(all_texts))                              # embed each distinct text once
            pos = {t: i for i, t in enumerate(texts)}
            inv = np.array([pos[t] for t in all_texts])
            cache = EMB_DIR / f"{name}-{sig}.npy"
            if cache.exists() and isinstance(results.get(name), dict) and "ingest_s" in results[name]:
                uvecs = np.load(cache)
                ingest_s = results[name]["ingest_s"]
            else:
                model.embed(texts[:8])                                  # warm-up
                probe = list(range(0, len(texts), max(1, len(texts) // 300)))[:300]
                t = time.time()
                model.embed([texts[i] for i in probe])
                projected = (time.time() - t) * len(texts) / len(probe)
                print(f"   projected ingest {projected / 60:.1f} min for {len(texts)} texts", flush=True)
                if projected > MAX_INGEST_S:
                    results[name] = {"model": mid, "skipped": f"projected ingest {projected / 60:.0f} min"}
                    continue
                t = time.time()
                uvecs = model.embed(texts)
                ingest_s = time.time() - t
                np.save(cache, uvecs)
            dvecs = normalize(uvecs[inv])
            variants = {"default": qp, **QUERY_VARIANTS.get(name, {})}
            for vname, prefix in variants.items():
                lat, qv = [], []
                model.embed([prefix + "warm up query"])
                for q in queries:
                    t = time.time()
                    qv.append(model.embed([prefix + q["q"]])[0])
                    lat.append(time.time() - t)
                qvecs = normalize(np.array(qv))
                dense, hybrid = evaluate(qvecs, dvecs, queries, rels, masks, bm_ranks)
                key = name if vname == "default" else f"{name} [{vname}]"
                results[key] = {"model": mid, "dim": int(dvecs.shape[1]), "load_s": round(load_s, 1),
                                "ingest_s": round(ingest_s, 1), "query_ms": round(1000 * statistics.median(lat), 1),
                                "dense": dense, "hybrid": hybrid}
                print(f"   {key}: ingest {ingest_s:.0f}s q {1000 * statistics.median(lat):.0f}ms "
                      f"dense {dense} hybrid {hybrid}", flush=True)
        except Exception as e:  # keep benchmarking the rest
            results[name] = {"model": mid, "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"   ERROR {results[name]['error']}", flush=True)
        finally:
            model = None
            gc.collect()
        res_path.write_text(json.dumps(results, indent=2))
    res_path.write_text(json.dumps(results, indent=2))
    write_markdown(results, len(chunks), len(queries))


def write_markdown(results, n_chunks, n_queries):
    rows = []
    for name, r in results.items():
        if "dense" not in r:
            rows.append(f"| {name} | — | — | — | — | — | — | {r.get('skipped') or r.get('error')} |")
            continue
        d, h = r["dense"], r["hybrid"]
        rows.append(f"| {name} | {r['dim']} | {r['ingest_s']:.0f} | {r['query_ms']:.0f} | "
                    f"{d['R@5']:.3f} / {d['R@10']:.3f} / {d['MRR@10']:.3f} | "
                    f"{h['R@5']:.3f} / {h['R@10']:.3f} / {h['MRR@10']:.3f} | "
                    f"{h['R@10'] / max(r['ingest_s'], 1) * 60:.3f} | |")
    md = [f"# Embedder benchmark ({n_chunks} chunks, {n_queries} queries)", "",
          "| model | dim | ingest s | query ms | dense R@5 / R@10 / MRR | hybrid R@5 / R@10 / MRR | hybrid R@10 per ingest-min | note |",
          "|---|---|---|---|---|---|---|---|", *rows]
    (OUT / "embedders.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
