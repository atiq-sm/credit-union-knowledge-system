"""Retrieval benchmark helpers: labels, BM25, fusion and metrics."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scoring import contains  # value-aware matching, identical to the official scorer

EVAL = Path(__file__).with_name("retrieval_eval.jsonl")


def load_queries():
    return [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]


def target_snapshot(as_of, dates):
    if not as_of:
        return len(dates) - 1
    ok = [i for i, d in enumerate(dates) if d <= as_of]
    return ok[-1] if ok else 0


def _path(u):
    return urlparse(u).path.rstrip("/").lower()


def relevant(query, chunks, dates):
    snap = target_snapshot(query.get("as_of"), dates)
    rel = {r.rstrip("/").lower() for r in query["rel"]}
    hits = set()
    for i, cc in enumerate(chunks):
        if snap not in cc.snapshots:
            continue
        c = cc.chunk
        if rel and _path(c.url) not in rel:
            continue
        hay = f"{c.title}\n{c.section}\n{c.text}"
        if any(all(contains(hay, t) for t in alt) for alt in query["ans"]):
            hits.add(i)
    return snap, hits


# BM25 lives in the submission so dev benchmarks and the solution share one copy
from bm25 import BM25, tokenize  # noqa: E402,F401


# ------------------------------------------------------------------------- ranking
def rank(scores: np.ndarray, mask: np.ndarray, k: int = 100) -> np.ndarray:
    s = np.where(mask, scores, -np.inf)
    k = min(k, int(mask.sum()))
    top = np.argpartition(-s, k - 1)[:k]
    return top[np.argsort(-s[top], kind="stable")]


def rrf(*rankings: np.ndarray, k: int = 60, n: int = 100) -> np.ndarray:
    acc: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            acc[int(i)] = acc.get(int(i), 0.0) + 1.0 / (k + pos + 1)
    return np.array(sorted(acc, key=lambda i: (-acc[i], i))[:n])


def metrics(ranked: list[np.ndarray], rels: list[set[int]]) -> dict:
    out = {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "R@20": 0.0, "MRR@10": 0.0}
    for r, rel in zip(ranked, rels):
        pos = next((p for p, i in enumerate(r[:20]) if int(i) in rel), None)
        for k in (1, 5, 10, 20):
            out[f"R@{k}"] += pos is not None and pos < k
        out["MRR@10"] += 1.0 / (pos + 1) if pos is not None and pos < 10 else 0.0
    n = len(ranked) or 1
    return {k: round(v / n, 4) for k, v in out.items()}
