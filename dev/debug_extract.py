"""Show retrieval hits and candidate sentence scores for one question."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import embedder as E, retrieval as RT, extract as X
chunks, dates = common.corpus()
emb = E.Embedder(log=lambda *_: None)
ret = RT.Retriever(chunks, dates, emb)
for q in sys.argv[1:]:
    hits = ret.search(q, len(dates) - 1)
    print("\n####", q)
    for ci, s, d in hits[:6]:
        c = ret.cc[ci]
        print(f"  hit {s:.3f} dense {d:.3f} {c.chunk.url.split('dupaco.com')[-1][:60]} | {c.chunk.section[:40]} | pub {max(c.published.values()) if c.published else '-'}")
    X.DEBUG = True
    print("  ANSWER:", X.answer_from_chunks(q, hits, ret, emb, len(dates) - 1)["text"][:300])
