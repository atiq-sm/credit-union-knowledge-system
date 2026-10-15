"""Ask the full solution arbitrary questions: .venv/Scripts/python dev/probe.py "q1" "q2" ..."""
import importlib.util, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
spec = importlib.util.spec_from_file_location("solution", common.ROOT / "submission" / "solution.py")
sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
ks = sol.build()
for s in common.snapshots():
    ks.ingest(s)
for q in sys.argv[1:]:
    t = time.time(); a = ks.answer(q); dt = time.time() - t
    hits = ks.retriever.search(q, len(ks.snapshots) - 1)
    print(f"- {q}  [{dt:.2f}s, top dense {hits[0][2]:.2f}, confident={a.confident}]\n    {a.text[:260]}")
