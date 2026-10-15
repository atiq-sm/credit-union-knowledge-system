"""
doc2query pilot: does matching questions against LLM-written "questions this chunk answers"
lift the gold chunk's retrieval rank?

    .venv/Scripts/python dev/d2q_pilot.py [n_miss] [n_ctrl]

1. Gold chunk per item = live chunk in the item's snapshot holding the most must_include
   tokens (>= half, no must_not token). Baseline rank from Retriever.search over all chunks.
2. Pilot = up to n_miss items with gold rank > 12 (retrieval misses) + n_ctrl items with
   gold rank <= 3 (controls, to catch harm).
3. Candidates = gold chunks + baseline top-15 of each pilot item (the competitors that
   matter). Local LLM (LM Studio, qwen2.5-7b-instruct) writes 5 questions per candidate;
   cached in dev/.cache/d2q.jsonl keyed by the chunk's embed key.
4. Re-rank with several fusions of chunk similarity and max generated-question similarity.
"""
import importlib.util
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from scoring import contains  # noqa: E402

LLM = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen2.5-7b-instruct"
CACHE = common.CACHE / "d2q.jsonl"
PROMPT = ("Passage from the Dupaco Community Credit Union website.\nPage title: {title}\nSection: {section}\n\n"
          "{text}\n\nWrite 5 different questions a member might ask that this passage answers. "
          "Use everyday wording, not the passage's wording. One question per line, no numbering.")


def gen_questions(chunk) -> list[str]:
    body = json.dumps({"model": MODEL, "temperature": 0.3, "max_tokens": 250, "messages": [{
        "role": "user", "content": PROMPT.format(title=chunk.title or "none", section=chunk.section or "none",
                                                 text=chunk.text[:2500])}]}).encode()
    req = urllib.request.Request(LLM, body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    qs = [l.strip().lstrip("-*0123456789.) ").strip() for l in out.splitlines()]
    return [q for q in qs if len(q) > 8 and q.endswith("?")][:6]


def load_items():
    files = [common.ROOT / "eval" / "public_eval.jsonl", *sorted((common.ROOT / "dev" / "probes").glob("*.jsonl")),
             *sorted((common.ROOT / "dev" / "heldout").glob("*.jsonl"))]
    items = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                it["_file"] = f.stem
                if it.get("category") != "should_refuse" and it.get("must_include"):
                    items.append(it)
    return items


def main():
    n_miss = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_ctrl = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    spec = importlib.util.spec_from_file_location("solution", common.ROOT / "submission" / "solution.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    from query import parse
    from embedder import doc_input, key
    ks = sol.build()
    for s in common.snapshots():
        ks.ingest(s)
    R, E = ks.retriever, ks.embedder
    N = len(R.cc)
    texts = [c.chunk.title + "\n" + c.chunk.section + "\n" + c.chunk.text for c in R.cc]

    rows = []
    for it in load_items():
        snap = ks.facts.snapshot_for(parse(it["question"], it.get("as_of")).as_of)
        must, mnot = it["must_include"], it.get("must_not_include", [])
        best, gold = 0, []
        for i in np.flatnonzero(R.masks[snap]):
            h = sum(contains(texts[i], t) for t in must)
            if h and h * 2 >= len(must) and not any(contains(texts[i], t) for t in mnot):
                if h > best:
                    best, gold = h, [int(i)]
                elif h == best:
                    gold.append(int(i))
        if not gold:
            continue
        qv = E.embed_query(it["question"])
        ranked = R.search(it["question"], snap, k=N, qvec=qv)
        pos = {ci: r for r, (ci, _, _) in enumerate(ranked)}
        rows.append(dict(it=it, snap=snap, gold=gold, rank=min(pos[g] for g in gold), qv=qv,
                         top=[ci for ci, _, _ in ranked[:15]], base={ci: (sc, d) for ci, sc, d in ranked[:300]}))
    print(f"{len(rows)} items with an identifiable gold chunk; "
          f"rank>12: {sum(r['rank'] > 12 for r in rows)}, rank<=3: {sum(r['rank'] <= 3 for r in rows)}")

    random.seed(0)
    miss = sorted([r for r in rows if r["rank"] > 12], key=lambda r: (r["it"]["_file"] not in ("h1", "h2"), -r["rank"]))
    miss = miss[:n_miss]
    ctrl = random.sample([r for r in rows if r["rank"] <= 3], min(n_ctrl, sum(r["rank"] <= 3 for r in rows)))
    pilot = miss + ctrl
    cand = sorted({ci for r in pilot for ci in r["top"] + r["gold"][:3]})

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["k"]] = d["qs"]
    todo = [ci for ci in cand if key(doc_input(R.cc[ci].chunk)).hex() not in cache]
    print(f"pilot {len(miss)} miss + {len(ctrl)} ctrl; {len(cand)} candidate chunks, {len(todo)} to generate")
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as f:
        for n, ci in enumerate(todo):
            k = key(doc_input(R.cc[ci].chunk)).hex()
            try:
                qs = gen_questions(R.cc[ci].chunk)
            except Exception as e:  # noqa: BLE001
                print("  gen failed:", e)
                continue
            cache[k] = qs
            f.write(json.dumps({"k": k, "url": R.cc[ci].chunk.url, "qs": qs}) + "\n")
            if n % 20 == 0:
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s")
    print(f"generation {time.time() - t0:.0f}s")

    # generated-question vectors, embedded like queries (question-to-question similarity)
    gq = {}
    for ci in cand:
        qs = cache.get(key(doc_input(R.cc[ci].chunk)).hex()) or []
        if qs:
            gq[ci] = np.array(list(E._model.embed(["task: search result | query: " + q for q in qs], batch_size=16)),
                              np.float32)
            gq[ci] /= np.linalg.norm(gq[ci], axis=1, keepdims=True)

    fusions = {
        "base": lambda sc, d, m: sc,
        "max(d,m)": lambda sc, d, m: sc - d + max(d, m),
        "d+0.3m": lambda sc, d, m: sc + 0.3 * m,
        "d+0.5m": lambda sc, d, m: sc + 0.5 * m,
        "0.5d+0.5m": lambda sc, d, m: sc - 0.5 * d + 0.5 * m,
    }
    res = {f: [] for f in fusions}
    for r in pilot:
        for fname, fn in fusions.items():
            scores = {}
            for ci, (sc, d) in r["base"].items():
                m = float((gq[ci] @ r["qv"]).max()) if ci in gq else d   # no questions: neutral
                scores[ci] = fn(sc, d, m)
            for g in r["gold"]:
                if g not in scores and g in gq:   # gold beyond top-300 baseline
                    d = float(R.D[g] @ r["qv"])
                    scores[g] = fn(d, d, float((gq[g] @ r["qv"]).max()))
            order = sorted(scores, key=lambda c: -scores[c])
            res[fname].append(min((order.index(g) for g in r["gold"] if g in scores), default=999))

    print("\nrank of gold chunk (0 = first)")
    print(f"{'id':10} {'kind':5} " + " ".join(f"{f:>10}" for f in fusions) + "  question")
    for i, r in enumerate(pilot):
        kind = "miss" if i < len(miss) else "ctrl"
        print(f"{r['it']['id']:10} {kind:5} " + " ".join(f"{res[f][i]:>10}" for f in fusions)
              + "  " + r["it"]["question"][:70])
    for kind, sl in (("miss", slice(0, len(miss))), ("ctrl", slice(len(miss), None))):
        print(f"\n{kind}: " + "  ".join(
            f"{f}: top3 {sum(x <= 2 for x in res[f][sl])} top12 {sum(x <= 11 for x in res[f][sl])} "
            f"mrr {np.mean([1 / (x + 1) for x in res[f][sl]]):.2f}" for f in fusions))
    # pool expansion: keep base top-12 order, append the top-3 chunks by generated-question match
    print(f"\ncorpus {N} chunks; live in current snapshot {int(R.masks[4].sum())}")
    print("pool expansion (base top12 + m-top3): gold in pool?")
    for i, r in enumerate(pilot):
        live = [ci for ci in gq if R.masks[r["snap"]][ci]]
        m = sorted(live, key=lambda c: -float((gq[c] @ r["qv"]).max()))[:3]
        pool = set(r["top"][:12]) | set(m)
        gin = any(g in pool for g in r["gold"])
        print(f"  {r['it']['id']:10} {'miss' if i < len(miss) else 'ctrl'}  base12 "
              f"{any(g in r['top'][:12] for g in r['gold'])!s:5} expanded {gin!s:5} "
              f"m-top sims {[round(float((gq[c] @ r['qv']).max()), 2) for c in m]}"
              f" gold m {max((round(float((gq[g] @ r['qv']).max()), 2) for g in r['gold'] if g in gq), default=None)}")
    # end to end: score answers with and without the expanded pool
    from run_eval import to_answer
    from scoring import score_item
    base_search = R.search

    def expanded_search(question, snap, k=8, qvec=None):
        hits = base_search(question, snap, k=k, qvec=qvec)
        qv = qvec if qvec is not None else E.embed_query(question)
        have = {ci for ci, _, _ in hits}
        live = [ci for ci in gq if R.masks[snap][ci] and ci not in have]
        extra = sorted(live, key=lambda c: -float((gq[c] @ qv).max()))[:3]
        return hits + [(ci, hits[-1][1], hits[-1][2]) for ci in extra]

    print("\nend to end (score base -> expanded)")
    tot = [0.0, 0.0]
    for i, r in enumerate(pilot):
        it = r["it"]
        out = []
        for fn in (base_search, expanded_search):
            R.search = fn
            ks._cache.clear()
            a = to_answer(ks.answer(it["question"], it.get("as_of")))
            out.append(score_item(it, a)["score"])
        R.search = base_search
        tot[0] += out[0]; tot[1] += out[1]
        print(f"  {it['id']:10} {'miss' if i < len(miss) else 'ctrl'}  {out[0]:.2f} -> {out[1]:.2f}")
    print(f"  total {tot[0]:.2f} -> {tot[1]:.2f}")
    for ci in [g for r in miss[:4] for g in r["gold"][:1]]:
        print("\nGOLD", R.cc[ci].chunk.url, "|", R.cc[ci].chunk.text[:120].replace("\n", " "))
        for q in cache.get(key(doc_input(R.cc[ci].chunk)).hex(), []):
            print("   ", q)


if __name__ == "__main__":
    main()
