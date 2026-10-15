"""
Knob sweep: scores every dev item (public + probes + h1-h7; h8 stays held out) with the
official scorer at each setting of module-level constants, against the defaults.

    .venv/Scripts/python dev/sweep.py extract.POOL_TOP=6,8 extract.PAGE_EXTRA=2,4 [--grid]

Each knob is varied alone (others at default) unless --grid, which tries every combination.
Prints weighted score, items won/lost against the baseline, and p95 answer latency.
"""
import importlib
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_eval import to_answer  # noqa: E402
from scoring import score_item  # noqa: E402

args = [a for a in sys.argv[1:] if a != "--grid"]
grid = "--grid" in sys.argv
knobs = []
for a in args:
    name, vals = a.split("=", 1)
    mod, attr = name.split(".")
    knobs.append((mod, attr, [json.loads(v) for v in vals.split(",")]))

files = [common.ROOT / "eval" / "public_eval.jsonl", *sorted((common.ROOT / "dev" / "probes").glob("*.jsonl")),
         *[common.ROOT / "dev" / "heldout" / f"h{i}.jsonl" for i in range(1, 8)]]
items = [json.loads(l) for f in files for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
W = {it["id"]: it.get("weight", 1.0) for it in items}

spec = importlib.util.spec_from_file_location("solution", common.ROOT / "submission" / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)
ks = sol.build()
for s in common.snapshots():
    ks.ingest(s)
mods = {m: importlib.import_module(m) for m, _, _ in knobs}
default = {(m, a): getattr(mods[m], a) for m, a, _ in knobs}


def run():
    ks._cache.clear()
    sc, lat = {}, []
    for it in items:
        t = time.time()
        a = to_answer(ks.answer(it["question"], it.get("as_of")))
        lat.append(time.time() - t)
        sc[it["id"]] = score_item(it, a)["score"]
    lat.sort()
    return sc, lat[int(.95 * len(lat))]


def total(sc):
    return sum(sc[i] * W[i] for i in sc) / sum(W.values())


base, p95 = run()
print(f"baseline {total(base):.4f} ({sum(v >= .999 for v in base.values())}/{len(items)} full) p95 {p95:.2f}s", flush=True)

if grid:
    settings = [list(zip([(m, a) for m, a, _ in knobs], combo)) for combo in itertools.product(*[v for *_, v in knobs])]
else:
    settings = [[((m, a), v)] for m, a, vs in knobs for v in vs]
for setting in settings:
    for (m, a), v in setting:
        setattr(mods[m], a, v)
    sc, p95 = run()
    won = sorted(i for i in sc if sc[i] > base[i] + 1e-6)
    lost = sorted(i for i in sc if sc[i] < base[i] - 1e-6)
    label = " ".join(f"{m}.{a}={v}" for (m, a), v in setting)
    print(f"{label}: {total(sc):.4f} ({total(sc) - total(base):+.4f})  won {len(won)} lost {len(lost)}  p95 {p95:.2f}s\n"
          f"   won  {won}\n   lost {lost}", flush=True)
    for (m, a), v in setting:
        setattr(mods[m], a, default[(m, a)])
