"""
Reranker pilot: blends a cross-encoder's score into the extractor's top candidate sentences
(extract.RERANK hook) and scores every dev item with the official scorer, per model and weight.

    .venv/Scripts/python dev/rerank_pilot.py MODEL[,MODEL...] [WEIGHTS=0.05,0.1,0.2] [TOP=12]

MODEL is a fastembed TextCrossEncoder name (e.g. Xenova/ms-marco-MiniLM-L-6-v2), or "laya:URL"
for a Laya HTTP server scoring the same pairs. Reports weighted score, items won/lost against
the baseline (no reranker) and reranker time per question. h7 is held out and never used here.
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_eval import to_answer  # noqa: E402
from scoring import score_item  # noqa: E402

args = sys.argv[1:]
models = args[0].split(",")
weights = [float(w) for w in (args[1] if len(args) > 1 else "0.05,0.1,0.2").split(",")]
top = int(args[2]) if len(args) > 2 else 12

files = [common.ROOT / "eval" / "public_eval.jsonl", *sorted((common.ROOT / "dev" / "probes").glob("*.jsonl")),
         *[common.ROOT / "dev" / "heldout" / f"h{i}.jsonl" for i in range(1, 7)]]
items = [json.loads(l) for f in files for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]

spec = importlib.util.spec_from_file_location("solution", common.ROOT / "submission" / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)
import extract  # noqa: E402

ks = sol.build()
for s in common.snapshots():
    ks.ingest(s)


def run() -> dict:
    ks._cache.clear()
    return {it["id"]: score_item(it, to_answer(ks.answer(it["question"], it.get("as_of"))))["score"] for it in items}


W = {it["id"]: it.get("weight", 1.0) for it in items}


def total(sc: dict) -> float:
    return sum(sc[i] * W[i] for i in sc) / sum(W.values())


base = run()
print(f"baseline {total(base):.4f} ({sum(v >= .999 for v in base.values())}/{len(items)} full)", flush=True)


def make_scorer(name: str):
    memo, spent = {}, [0.0, 0]
    if name.startswith("laya:"):
        import urllib.request
        url = name[5:]

        def raw(q, sents):
            body = json.dumps({"question": q, "sentences": sents}).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())["scores"]
    else:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        enc = TextCrossEncoder(name, cache_dir=str(common.ROOT / "dev" / ".cache" / "rerankers"))

        def raw(q, sents):
            return [1 / (1 + math.exp(-x)) for x in enc.rerank(q, sents)]   # logits -> 0..1

    def score(q, sents):
        new = [s for s in sents if (q, s) not in memo]
        if new:
            t = time.time()
            for s, v in zip(new, raw(q, new)):
                memo[(q, s)] = v
            spent[0] += time.time() - t
            spent[1] += 1
        return [memo[(q, s)] for s in sents]
    return score, spent


extract.RERANK_TOP = top
for name in models:
    scorer, spent = make_scorer(name)
    extract.RERANK = scorer
    for w in weights:
        extract.RERANK_WEIGHT = w
        sc = run()
        won = sorted(i for i in sc if sc[i] > base[i] + 1e-6)
        lost = sorted(i for i in sc if sc[i] < base[i] - 1e-6)
        ms = 1000 * spent[0] / max(spent[1], 1)
        print(f"{name} w={w} top={top}: {total(sc):.4f}  won {len(won)} lost {len(lost)}  "
              f"rerank {ms:.0f} ms/call (first pass)", flush=True)
        print(f"   won  {won}\n   lost {lost}", flush=True)
extract.RERANK = None
