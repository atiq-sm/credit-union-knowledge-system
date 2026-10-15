"""
Dump the fact table for review: per key, the resolved value in each snapshot plus any
other value seen for that key elsewhere (stale pages, older snapshots).

    .venv/Scripts/python dev/dump_facts.py [family-substring ...] > dev/.cache/facts.txt
"""
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from facts import build_fact_table, page_path  # noqa: E402

chunks, dates = common.corpus()
t = time.time()
ft = build_fact_table(chunks, dates)
print(f"# {len(ft.obs)} keys, {sum(len(v) for v in ft.obs.values())} observations, built in {time.time() - t:.2f}s")
fams = collections.Counter((k.scope, k.family) for k in ft.obs)
print("# keys per family:", dict(sorted(fams.items())))
want = sys.argv[1:]
for key in sorted(ft.obs, key=lambda k: (k.scope, k.family, k.metric, str(k.attrs))):
    if want and not any(w in key.family for w in want):
        continue
    if key.metric in ("min_open", "min_balance", "max_balance", "points", "payment"):
        continue
    tl = ft.timeline(key)
    vals = " | ".join(f"{o.text}@{page_path(o.url)[:18]}" if o else "-" for o in tl)
    print(f"\n{key.describe()}  [{key.scope}/{key.family}]\n   resolved: {vals}")
    others = collections.defaultdict(set)
    for o in ft.obs[key]:
        best = tl[o.snapshot]
        if best and o.text != best.text:
            others[(o.text, page_path(o.url))].add(o.snapshot + 1)
    for (v, u), ss in sorted(others.items()):
        print(f"   other: {v} on {u} in snapshots {''.join(map(str, sorted(ss)))}")
