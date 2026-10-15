"""
Print the conflicts the solution would report, best first.

    .venv/Scripts/python dev/show_conflicts.py [-v] [substring ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from conflicts import all_conflicts  # noqa: E402
from orgfacts import OrgFacts  # noqa: E402
from claims import Claims  # noqa: E402
from facts import build_fact_table, page_path  # noqa: E402

verbose = "-v" in sys.argv
want = [a.lower() for a in sys.argv[1:] if a != "-v"]
chunks, dates = common.corpus()
ft = build_fact_table(chunks, dates)
found = all_conflicts(ft, OrgFacts(chunks), Claims(chunks), chunks, dates)
print(f"{len(found)} conflicts")
for f in found:
    if want and not any(w in f.topic.lower() for w in want):
        continue
    print(f"\n[{f.weight:.1f}] {f.topic}\n   values: {f.values}\n   resolution: {f.resolution}")
    if verbose:
        print(f"   reason: {f.reason}")
        print("   sources: " + ", ".join(f"{page_path(u)}@{s + 1}" for u, s, _ in f.sources))
