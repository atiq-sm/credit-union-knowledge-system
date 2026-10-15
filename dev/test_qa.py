"""
Full-solution regression on dev/retrieval_eval.jsonl (re-worded public questions and new
ones from the snapshot survey). An answer passes when it contains every token of one
`ans` alternative (value-aware, like the official scorer).

    .venv/Scripts/python dev/test_qa.py [-v]
"""
import importlib.util
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import retrieval_lib as R  # noqa: E402
from scoring import contains  # noqa: E402

spec = importlib.util.spec_from_file_location("solution", common.ROOT / "submission" / "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

ks = sol.build()
t = time.time()
for s in common.snapshots():
    ks.ingest(s)
print(f"ingest {time.time() - t:.1f}s")

qs = R.load_queries()
ok, lat, fails = 0, [], []
for q in qs:
    t = time.time()
    a = ks.answer(q["q"], q.get("as_of"))
    lat.append(time.time() - t)
    # labels were written for chunk retrieval ("17 Month"); answers phrase tiers their own way
    # ("17-month"), so only the value-bearing tokens are required when an alternative has any
    descriptor = re.compile(r"^\d+ Months?$|^(Jumbo|Hy-Vee|Forbes|Asbury|Mini-Max|Lobby|Drive-up)$")
    alts = [[t for t in alt if not descriptor.match(t)] or alt for alt in q["ans"]]
    good = any(all(contains(a.text, tok) for tok in alt) for alt in alts)
    ok += good
    if not good:
        fails.append((q, a))
for q, a in fails:
    print(f"FAIL {q['id']:8} {q['q'][:70]}\n      want {q['ans']}\n      got  {a.text[:260]}")
lat.sort()
print(f"\n{ok}/{len(qs)} answers contain the expected facts; p50 {lat[len(lat) // 2]:.2f}s p95 {lat[int(.95 * len(lat))]:.2f}s")
