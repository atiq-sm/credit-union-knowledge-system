"""
Found-vs-picked diagnosis: for every item that loses points, say *where* it was lost.

    .venv/Scripts/python dev/diagnose.py [files...] [-k substr] [-v]

Default files: eval/public_eval.jsonl, dev/probes/*.jsonl, dev/heldout/*.jsonl.

Per item: the answer route (rates, outline, org, claims, branches, extract...), the best
"gold" chunk (live in the item's snapshot, holding the most must_include tokens, no
must_not token) and its retrieval rank, whether a gold sentence made the extractor's
candidate pool, and the extractor's own ranking of it. Classes, first match wins:

  TRAP            a must_not_include token is in the answer (item scores 0)
  REFUSE_MISS     should_refuse item answered confidently / with new numbers
  FALSE_REFUSAL   answerable item refused
  ROUTE           a rule route (rates/org/claims/branches/outline) answered and missed tokens
  NO_GOLD         no single chunk holds half the tokens (spread, paraphrased, or absent)
  NOT_RETRIEVED   gold chunk below rank 12
  NOT_POOLED      gold chunk in top 12 but not in the extractor's pool (top 4 + phrase hits)
  SPREAD          pool holds the tokens, but no single sentence holds them all
  PICKED_WRONG    a pool sentence holds the tokens, the extractor chose another

Writes dev/.cache/diagnose.txt and dev/.cache/diagnose.jsonl.
"""
import contextlib
import importlib.util
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_eval import to_answer  # noqa: E402
from scoring import REFUSAL_RE, contains, score_item  # noqa: E402

args = sys.argv[1:]
verbose = "-v" in args
key = args[args.index("-k") + 1] if "-k" in args else None
files = [Path(a) for a in args if a.endswith(".jsonl")] or [
    common.ROOT / "eval" / "public_eval.jsonl", *sorted((common.ROOT / "dev" / "probes").glob("*.jsonl")),
    *[common.ROOT / "dev" / "heldout" / f"h{i}.jsonl" for i in (1, 2)]]
items = []
for f in files:
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            it = json.loads(line)
            it["_file"] = f.stem
            if not key or key in it["id"]:
                items.append(it)

spec = importlib.util.spec_from_file_location("solution", common.ROOT / "submission" / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)
import extract  # noqa: E402  (same module object solution imported)
from corpus import strip_links  # noqa: E402
from query import parse  # noqa: E402

ks = sol.build()
for s in common.snapshots():
    ks.ingest(s)
R = ks.retriever

# ---- route tracing: wrap every answer source, record (name, text) of non-None results
calls: list[tuple[str, str]] = []


def traced(name, fn):
    def w(*a, **kw):
        r = fn(*a, **kw)
        if r is not None:
            calls.append((name, r.get("text", "")))
        return r
    return w


sol.outline_answer = traced("outline", sol.outline_answer)
sol.answer_rate = traced("rates", sol.answer_rate)
_afc = sol.answer_from_chunks
sol.answer_from_chunks = traced("extract", _afc)
ks.org.answer = traced("org", ks.org.answer)
ks.claims.answer = traced("claims", ks.claims.answer)
_br = ks._branches


def branches(snap):
    b = _br(snap)
    if not getattr(b, "_traced", False):
        b.answer = traced("branches", b.answer)
        b._traced = True
    return b


ks._branches = branches


def route_of(text: str) -> str:
    names = [n for n, t in calls if t and t in text]
    if not names:
        return calls[-1][0] if calls else "?"
    # "rates + extract" when the fine print was appended
    return "+".join(dict.fromkeys(names))


def toks_in(text, toks):
    return [t for t in toks if contains(text, t)]


def pool(question, hits, snap):
    out = []
    seen = set()
    pl = hits[:4] + extract._phrase_hits(question, hits, R)
    if getattr(extract, "KEYWORD_EXTRA", 0):
        pl += R.keyword_hits(question, snap, {h[0] for h in pl}, k=extract.KEYWORD_EXTRA,
                             qvec=ks.embedder.embed_query(question))
    for rank, (ci, score, dense) in enumerate(pl):
        for s, _pos, _lst in extract._sentences(R.cc[ci].chunk.text):
            plain = strip_links(s)
            if (len(plain) < 25 and not re.search(r"\d", plain)) or plain.lower() in seen:
                continue
            seen.add(plain.lower())
            out.append((plain, rank, ci))
    return out


DEBUG_LINE = re.compile(r"^\s+(-?\d+\.\d+) sim (-?\d+\.\d+) r(\d+) (.*)$")
rows = []
for it in items:
    q, as_of = it["question"], it.get("as_of")
    cat = it.get("category", "")
    must, mnot = it.get("must_include", []), it.get("must_not_include", [])
    calls.clear()
    ks._cache.clear()
    extract.DEBUG = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a = to_answer(ks.answer(q, as_of))
    extract.DEBUG = False
    sc = score_item(it, a)
    text = a.text or ""
    route = route_of(text)
    dbg = [m.groups() for m in map(DEBUG_LINE.match, buf.getvalue().splitlines()) if m]
    row = dict(id=it["id"], file=it["_file"], cat=cat, q=q, as_of=as_of, score=sc["score"],
               corr=sc["correctness"], route=route, answer=text[:300], must=must, mnot=mnot)
    if sc["correctness"] >= 1.0:
        row["cls"] = "OK"
        rows.append(row)
        continue
    refused = (not a.confident) or bool(REFUSAL_RE.search(text))
    trap = toks_in(text, mnot)
    missing = [t for t in must if not contains(text, t)]
    row["missing"] = missing
    snap = ks.facts.snapshot_for(parse(q, as_of).as_of)

    # gold chunk + retrieval rank
    texts = {}
    best, gold = 0, []
    for i in np.flatnonzero(R.masks[snap]):
        c = R.cc[i].chunk
        t = c.title + "\n" + c.section + "\n" + c.text
        h = len(toks_in(t, must)) if must else 0
        if h and h * 2 >= len(must) and not toks_in(t, mnot):
            texts[int(i)] = t
            if h > best:
                best, gold = h, [int(i)]
            elif h == best:
                gold.append(int(i))
    hits = R.search(q, snap, k=12)
    if gold:
        ranked = R.search(q, snap, k=int(R.masks[snap].sum()))
        pos = {ci: r for r, (ci, _, _) in enumerate(ranked)}
        g = min(gold, key=lambda c: pos[c])
        row.update(gold_rank=pos[g], gold_url=R.cc[g].chunk.url, gold_hits=best,
                   gold_text=R.cc[g].chunk.text[:200].replace("\n", " "))
    P = pool(q, hits, snap)
    sent_best = max((len(toks_in(s, must)), -n) for n, (s, _, _) in enumerate(P)) if P and must else (0, 0)
    pool_union = toks_in(" ".join(s for s, _, _ in P), must) if must else []
    row["pool_size"] = len(P)
    row["pool_sentence_hits"] = sent_best[0]
    row["pool_union_hits"] = len(pool_union)
    if sent_best[0]:
        gs = P[-sent_best[1]][0]
        row["gold_sentence"] = gs[:200]
        # where the extractor ranked it (DEBUG prints its top 8)
        rk = next((n for n, d in enumerate(dbg) if gs[:60] in d[3] or d[3][:60] in gs), None)
        row["gold_sentence_rank"] = rk if rk is not None else ">8"
        if dbg:
            row["winner_val"] = float(dbg[0][0])
            if rk is not None:
                row["gold_val"] = float(dbg[rk][0])

    if trap:
        cls = "TRAP"
    elif cat == "should_refuse":
        cls = "REFUSE_MISS"
    elif refused:
        cls = "FALSE_REFUSAL"
    elif not route.startswith("extract") and route != "?":
        cls = "ROUTE"
    elif not gold:
        cls = "NO_GOLD"
    elif row["gold_rank"] >= 12:
        cls = "NOT_RETRIEVED"
    elif not any(ci in gold for _, _, ci in P) and sent_best[0] < len(must):
        cls = "NOT_POOLED"
    elif sent_best[0] < len(must) and len(pool_union) > sent_best[0]:
        cls = "SPREAD"
    elif sent_best[0] > len(must) - len(missing):
        cls = "PICKED_WRONG"
    else:
        cls = "NOT_POOLED"
    row["cls"] = cls
    if cls == "TRAP":
        row["trap"] = trap
    if cls in ("FALSE_REFUSAL", "ROUTE", "TRAP"):
        # would plain extraction have had the answer?
        row["gold_in_top12"] = bool(gold) and row.get("gold_rank", 99) < 12
    rows.append(row)

out = []
fails = [r for r in rows if r["cls"] != "OK"]
weight = {it["id"]: it.get("weight", 1.0) for it in items}
for r in sorted(fails, key=lambda r: (r["cls"], r["file"], r["id"])):
    lost = (1 - r["score"]) * weight[r["id"]]
    out.append(f"[{r['cls']}] {r['id']} ({r['file']}, {r['cat']}) score {r['score']:.2f} lost {lost:.2f} "
               f"route={r['route']}")
    out.append(f"   Q: {r['q']}" + (f"  (as_of {r['as_of']})" if r["as_of"] else ""))
    out.append(f"   A: {r['answer'][:220]}")
    out.append(f"   missing {r.get('missing')}" + (f"  TRAP {r['trap']}" if r.get("trap") else ""))
    if "gold_rank" in r:
        out.append(f"   gold rank {r['gold_rank']} ({r['gold_hits']}/{len(r['must'])} tok) {r['gold_url']}")
        out.append(f"        {r['gold_text'][:160]}")
    if r.get("gold_sentence"):
        out.append(f"   pool sentence ({r['pool_sentence_hits']}/{len(r['must'])} tok, extractor rank "
                   f"{r['gold_sentence_rank']}, val {r.get('gold_val', '?')} vs winner {r.get('winner_val', '?')}):"
                   f" {r['gold_sentence'][:160]}")
    if "gold_in_top12" in r:
        out.append(f"   gold in top12: {r['gold_in_top12']}")
    out.append("")

summary_at = len(out)
n = len(rows)
lost_by = defaultdict(float)
cnt = Counter()
for r in fails:
    lost_by[r["cls"]] += (1 - r["score"]) * weight[r["id"]]
    cnt[r["cls"]] += 1
tot_w = sum(weight[r["id"]] for r in rows)
out.append(f"{n} items, {n - len(fails)} full marks, weighted score {sum(r['score'] * weight[r['id']] for r in rows) / tot_w:.3f}")
out.append("points lost by class (weighted, share of all lost):")
tl = sum(lost_by.values()) or 1
for c, v in sorted(lost_by.items(), key=lambda x: -x[1]):
    out.append(f"  {c:15} {cnt[c]:3} items  lost {v:6.2f}  {100 * v / tl:4.0f}%")
rl = defaultdict(float)
for r in fails:
    rl[(r["cls"], r["route"])] += (1 - r["score"]) * weight[r["id"]]
out.append("by class x route:")
for (c, rt), v in sorted(rl.items(), key=lambda x: -x[1])[:15]:
    out.append(f"  {c:15} {rt:22} {v:6.2f}")
fl = defaultdict(float)
for r in fails:
    fl[r["file"]] += (1 - r["score"]) * weight[r["id"]]
out.append("lost by file: " + ", ".join(f"{f} {v:.1f}" for f, v in sorted(fl.items(), key=lambda x: -x[1])))

report = "\n".join(out)
(common.CACHE / "diagnose.txt").write_text(report, encoding="utf-8")
with (common.CACHE / "diagnose.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, default=str) + "\n")
print(report if verbose else "\n".join(out[summary_at:]))
