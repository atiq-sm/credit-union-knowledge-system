"""
Refreshes submission/data/sent_cache.npz: vectors of every candidate sentence the extractor
can embed at answer time (extract.sentence_inputs over the whole corpus), so answering
rarely runs the model. Vectors already in the old file are kept; missing ones are embedded
with the solution's own embedder; sentences no longer produced are dropped.

    .venv/Scripts/python dev/build_sent_cache.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import embedder as E  # noqa: E402  (submission/embedder.py)
from extract import sentence_inputs  # noqa: E402

chunks, _ = common.corpus(rebuild=True)
inputs = sorted({t for c in chunks for t in sentence_inputs(c.chunk)})
emb = E.Embedder()
old = emb.sent_vecs
todo = sorted((t for t in inputs if E.key(t) not in old), key=len)
print(f"{len(inputs)} unique sentence inputs, {len(todo)} not in the shipped cache")
t0 = time.time()
new = {}
for k in range(0, len(todo), 64):
    batch = todo[k:k + 64]
    vecs = np.array(list(emb._model.embed(batch, batch_size=16)), np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    new.update((E.key(t), v) for t, v in zip(batch, vecs))
    if k % 1280 == 0:
        print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
keys = np.array([np.frombuffer(E.key(t), np.uint8) for t in inputs])
vecs = np.stack([old.get(E.key(t), new.get(E.key(t))) for t in inputs]).astype(np.float16)
np.savez(E.SENT_CACHE_FILE, model=np.array(E.MODEL_ID), keys=keys, vecs=vecs)
print(f"wrote {len(inputs)} vectors to {E.SENT_CACHE_FILE} ({E.SENT_CACHE_FILE.stat().st_size / 1e6:.1f} MB) "
      f"in {time.time() - t0:.0f}s")
