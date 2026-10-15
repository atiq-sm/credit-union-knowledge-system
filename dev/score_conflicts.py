"""
Score conflicts() against a gold list without running the QA pass, and show which
submitted entry each gold entry matched.

    .venv/Scripts/python dev/score_conflicts.py [gold.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from scoring import _norm, contains, score_conflicts  # noqa: E402
from conflicts import all_conflicts  # noqa: E402
from facts import build_fact_table  # noqa: E402
from orgfacts import OrgFacts  # noqa: E402
from claims import Claims  # noqa: E402
from types import SimpleNamespace  # noqa: E402

gold = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else common.ROOT / "dev" / "conflicts_gold_dev.json").read_text())
chunks, dates = common.corpus()
found = all_conflicts(build_fact_table(chunks, dates), OrgFacts(chunks), Claims(chunks), chunks, dates)
subs = [SimpleNamespace(topic=f.topic, values=f.values, resolution=f.resolution) for f in found]
print(json.dumps(score_conflicts(gold, subs)))
used = set()
for g in gold:
    hit = None
    for i, s in enumerate(subs):
        if i in used:
            continue
        vals = " | ".join(s.values)
        vh = sum(contains(vals, v) for v in g["values"])
        kh = any(_norm(k) in _norm(s.topic) for k in g.get("topic_keywords", []))
        if vh >= 2 or (vh >= 1 and kh):
            hit = i
            used.add(i)
            break
    res = "" if hit is None else ("RES-OK" if g.get("correct_value") and contains(subs[hit].resolution or "", g["correct_value"]) else "res-miss")
    print(f"{g['id']} {str(g['values']):40} -> {'MISS' if hit is None else f'#{hit} {subs[hit].topic[:70]}'} {res}")
print("unmatched submissions:")
for i, s in enumerate(subs):
    if i not in used:
        print(f"  #{i} {s.topic[:100]}")
