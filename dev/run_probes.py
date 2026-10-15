"""
Runs probe questions (public_eval format) through the full solution and scores each one
with the official scorer. Prints every failure with the answer, the trap or missed tokens
and the probe's note, then a score per file and per category.

    .venv/Scripts/python dev/run_probes.py [files...] [-v] [-k substr]

Default files: dev/probes/*.jsonl. -v prints passing items too; -k keeps ids containing substr.
"""
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from run_eval import to_answer  # noqa: E402
from scoring import score_item  # noqa: E402

args = [a for a in sys.argv[1:]]
verbose = "-v" in args
key = args[args.index("-k") + 1] if "-k" in args else None
files = [Path(a) for a in args if a.endswith(".jsonl")] or sorted((common.ROOT / "dev" / "probes").glob("*.jsonl"))
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
ks = sol.build()
t = time.time()
for s in common.snapshots():
    ks.ingest(s)
print(f"ingest {time.time() - t:.1f}s, {len(items)} probes")

by_file, by_cat, lat, out = defaultdict(list), defaultdict(list), [], []
for it in items:
    t = time.time()
    a = to_answer(ks.answer(it["question"], it.get("as_of")))
    lat.append(time.time() - t)
    r = score_item(it, a)
    by_file[it["_file"]].append(r["score"])
    by_cat[it["category"]].append(r["score"])
    out.append({"id": it["id"], "score": r["score"], "answer": a.text, "confident": a.confident,
                "sources": [s.url for s in a.sources][:3]})
    if r["correctness"] < 0.999 or verbose:
        tag = "ok  " if r["correctness"] >= 0.999 else "FAIL"
        print(f"\n{tag} {it['id']} [{it['category']}] as_of={it.get('as_of')} :: {it['question']}")
        print(f"   want {it.get('must_include')}  not {it.get('must_not_include')}  "
              f"missed {r['missed']} trap {r['trap_hits']}")
        print(f"   exp  {it.get('expected_answer', '')[:300]}")
        print(f"   got  {a.text[:500]}  (confident={a.confident})")
        if r["correctness"] < 0.999:
            print(f"   note {it.get('note', '')[:400]}")

(common.ROOT / "dev" / ".cache").mkdir(parents=True, exist_ok=True)
(common.ROOT / "dev" / ".cache" / "probe_answers.jsonl").write_text(
    "\n".join(json.dumps(o) for o in out), encoding="utf-8")
print("\n== by file")
for k, v in sorted(by_file.items()):
    print(f"  {k:12} {sum(v) / len(v):.3f}  ({sum(x >= .999 for x in v)}/{len(v)} full)")
print("== by category")
for k, v in sorted(by_cat.items()):
    print(f"  {k:20} {sum(v) / len(v):.3f}  ({sum(x >= .999 for x in v)}/{len(v)} full)")
lat.sort()
print(f"p50 {lat[len(lat) // 2]:.2f}s p95 {lat[int(.95 * len(lat))]:.2f}s")
