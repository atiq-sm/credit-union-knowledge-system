"""
Refreshes submission/data/emb_cache.npz so the solution's ingest embeds nothing: every
model input of the current corpus is looked up in the shipped cache, missing ones are
embedded with the solution's own embedder (same fastembed EmbeddingGemma-300m on CPU),
and inputs no longer in the corpus are dropped.

    .venv/Scripts/python dev/build_emb_cache.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import embedder as E  # noqa: E402  (submission/embedder.py)

chunks, _ = common.corpus(rebuild=True)
inputs = sorted(set(E.doc_input(c.chunk) for c in chunks))
emb = E.Embedder()
missing = sum(E.key(t) not in emb.cache for t in inputs)
print(f"{len(inputs)} unique model inputs, {missing} not in the shipped cache")
t = time.time()
emb.embed_docs(inputs)
print(f"embedded the missing ones in {time.time() - t:.0f}s")
keys = np.array([np.frombuffer(E.key(t), np.uint8) for t in inputs])
vecs = np.stack([emb.cache[E.key(t)] for t in inputs])
E.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez(E.CACHE_FILE, model=np.array(E.MODEL_ID), keys=keys, vecs=vecs.astype(np.float16))
print(f"wrote {len(inputs)} vectors to {E.CACHE_FILE} ({E.CACHE_FILE.stat().st_size / 1e6:.1f} MB)")
