"""
Dense embeddings with EmbeddingGemma-300m (ONNX, CPU, via fastembed).

Chosen by dev/bench_embedders.py: best retrieval quality on the snapshot corpus
(dense R@5 0.94 / MRR@10 0.79 vs 0.81 / 0.59 for bge-small), but ~17 min to embed
the full corpus on a laptop CPU. Three guards keep ingest well inside the 30-minute limit:

  1. A shipped vector cache (data/emb_cache.npz, built by dev/build_emb_cache.py)
     keyed by a hash of the exact model input. Chunks seen before cost nothing.
  2. A time budget for embedding cache misses. Chunks left over when it runs out get
     no vector; retrieval falls back to BM25 for them.
  3. If the model cannot be loaded at all (no network, missing package) the embedder
     reports itself unavailable and the system runs on BM25 alone.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-300m"
QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: {title} | text: "
DIM = 768
HERE = Path(__file__).resolve().parent
CACHE_FILE = HERE / "data" / "emb_cache.npz"
SENT_CACHE_FILE = HERE / "data" / "sent_cache.npz"   # candidate sentences (dev/build_sent_cache.py)
MODEL_DIR = Path(os.getenv("KS_MODEL_DIR", HERE / ".model_cache"))
EMBED_BUDGET_S = float(os.getenv("KS_EMBED_BUDGET_S", 18 * 60))
BATCH = 8           # small, length-sorted batches: least padding, fastest on CPU


def doc_input(chunk) -> str:
    """Exact string fed to the model for a corpus.Chunk (also its cache key)."""
    from corpus import strip_links
    return (DOC_PREFIX.format(title=chunk.title or "none")
            + (f"{chunk.section}\n" if chunk.section else "") + strip_links(chunk.text))


def key(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Embedder:
    def __init__(self, cache_file: Path = CACHE_FILE, log=print):
        self.log = log
        self.cache: dict[bytes, np.ndarray] = {}
        if cache_file.exists():
            data = np.load(cache_file)
            if str(data["model"]) == MODEL_ID:
                self.cache = dict(zip((bytes(k) for k in data["keys"]), data["vecs"].astype(np.float32)))
        self._model = None
        self._sent_cache: dict[str, np.ndarray] = {}
        self.sent_vecs: dict[bytes, np.ndarray] = {}         # shipped, fp16, keyed like the chunk cache
        if SENT_CACHE_FILE.exists():
            data = np.load(SENT_CACHE_FILE)
            if str(data["model"]) == MODEL_ID:
                self.sent_vecs = dict(zip((bytes(k) for k in data["keys"]), data["vecs"]))
        self.available = True
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(MODEL_ID, cache_dir=str(MODEL_DIR))
        except Exception as e:  # no network / package: keep running on BM25 only
            self.available = False
            self.log(f"embedder unavailable ({type(e).__name__}: {e}); using BM25 only")

    def embed_docs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors for `texts` and a mask of which rows actually have a vector."""
        out = np.zeros((len(texts), DIM), np.float32)
        have = np.zeros(len(texts), bool)
        todo = []
        for i, t in enumerate(texts):
            v = self.cache.get(key(t))
            if v is not None:
                out[i], have[i] = v, True
            else:
                todo.append(i)
        if todo and self._model is not None:
            todo.sort(key=lambda i: len(texts[i]))
            t0 = time.time()
            for k in range(0, len(todo), BATCH):
                if time.time() - t0 > EMBED_BUDGET_S:
                    self.log(f"embedding budget spent; {len(todo) - k} chunks left to BM25 only")
                    break
                idx = todo[k:k + BATCH]
                vecs = np.array(list(self._model.embed([texts[i] for i in idx], batch_size=BATCH)), np.float32)
                out[idx], have[idx] = vecs, True
                for i, v in zip(idx, vecs):
                    self.cache[key(texts[i])] = v
        self.log(f"embedded {int(have.sum())}/{len(texts)} chunks ({len(texts) - len(todo)} from cache)")
        return _normalize(out) * have[:, None], have

    def embed_docs_uncached(self, texts: list[str]) -> np.ndarray:
        """Unit vectors for short answer-time texts (candidate sentences). Kept in memory: paraphrases
        and repeated questions read the same passages, and a reused vector also keeps answers stable."""
        if self._model is None or not texts:
            return np.zeros((len(texts), DIM), np.float32)
        for t in texts:
            if t not in self._sent_cache:
                v = self.sent_vecs.get(key(t))
                if v is not None:
                    self._sent_cache[t] = v.astype(np.float32)
        todo = sorted({t for t in texts if t not in self._sent_cache}, key=len)
        if todo:
            vecs = _normalize(np.array(list(self._model.embed(todo, batch_size=16)), np.float32))
            if len(self._sent_cache) > 200_000:
                self._sent_cache.clear()
            self._sent_cache.update(zip(todo, vecs))
        return np.stack([self._sent_cache[t] for t in texts])

    def embed_query(self, question: str) -> np.ndarray | None:
        if self._model is None:
            return None
        key = QUERY_PREFIX + question
        if key not in self._sent_cache:
            v = np.array(next(iter(self._model.embed([key], batch_size=1))), np.float32)
            self._sent_cache[key] = _normalize(v)
        return self._sent_cache[key]
