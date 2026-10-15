"""Grid over retrieval priors (authority / recency weights) on the dev retrieval set."""
import sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, retrieval_lib as R
import embedder as E, retrieval as RT
chunks, dates = common.corpus()
emb = E.Embedder(log=lambda *_: None)
ret = RT.Retriever(chunks, dates, emb)
qs = R.load_queries(); rels = [R.relevant(q, chunks, dates)[1] for q in qs]
qv = [emb.embed_query(q["q"]) for q in qs]
snaps = [R.target_snapshot(q.get("as_of"), dates) for q in qs]
for wa, wr in itertools.product((0.0, 0.03, 0.06), (0.0, 0.02, 0.04, 0.08)):
    RT.W_AUTH, RT.W_RECENT = wa, wr
    ranks = [np.array([i for i, *_ in ret.search(q["q"], snaps[n], k=20, qvec=qv[n])]) for n, q in enumerate(qs)]
    m = R.metrics(ranks, rels)
    miss = [qs[n]["id"] for n in range(len(qs)) if not any(int(i) in rels[n] for i in ranks[n][:5])]
    print(f"auth={wa:.2f} recent={wr:.2f}  R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f} MRR {m['MRR@10']:.3f}  miss@5 {miss}")
