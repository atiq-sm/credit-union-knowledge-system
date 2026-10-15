# A knowledge system that knows what's true now

My solution to the interface.ai knowledge-system build assignment. It answers credit-union
members' questions from five crawls of the credit union's website (May–Sep 2026). It gives the
value that is current (or current on a given date), in the right tier, with a source, and it
says "I don't know" when the site doesn't say. It also lists where the site contradicts itself,
with the value it trusts and why.

**The solution is in [`submission/`](submission/).** [`submission/NOTES.md`](submission/NOTES.md)
explains the approach: how "current" is decided, the answer routes, and the results.
No API keys and no LLM are needed. Everything runs on CPU.

## Layout

```
submission/        the graded folder: solution.py with build(), requirements.txt, NOTES.md
  data/            shipped embedding caches (hashes + vectors only, no site text)
dev/               my tooling: regression tests, probe and held-out question sets, diagnostics
harness/, eval/    the kit's scorer and 40 practice questions, unchanged
starter/           the kit's naive starter, unchanged
```

The website snapshots (`snapshots/`, `snapshots_zip/`) are left out on purpose, because the
data must not be redistributed. Put the kit's `snapshots_zip/` folder in the repo root, and
the scorer unpacks it on first run.

## Run

```
python -m venv .venv
.venv/Scripts/pip install -r submission/requirements.txt     # Windows; use .venv/bin/pip elsewhere
.venv/Scripts/python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

The first run downloads the EmbeddingGemma-300m ONNX model (about 1.2 GB) from Hugging Face
into `submission/.model_cache/`.

## Results (practice set)

| Pillar | Score |
|---|---|
| Accuracy | 0.967 (ceiling 0.969 without the LLM judge) |
| Conflicts | 0.930 |
| Consistency | 1.0 |
| Speed | 1.0 (p95 0.04 s) |
| **Final** | **0.961** |

The practice set is what the system was tuned on, so 0.967 is optimistic. Fresh questions
written by agents that never saw the code scored 0.58–0.74 on their first runs. `NOTES.md`
covers these numbers and what each round of fixes changed.
