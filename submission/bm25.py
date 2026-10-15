"""Okapi BM25 over chunk text (keyword fallback when dense vectors are unavailable)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOK = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOP = set("a an and are as at be by can do does for from has have how i in is it its me my of on or our "
            "the this to was what when where which who will with you your dupaco".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [tokenize(d) for d in docs]
        self.len = np.array([len(t) for t in toks], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(toks) else 1.0
        self.post: dict[str, list[tuple[int, int]]] = {}
        for i, t in enumerate(toks):
            for w, n in Counter(t).items():
                self.post.setdefault(w, []).append((i, n))
        self.N = len(docs)
        self.idf = {w: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.post.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        for w in set(tokenize(query)):
            for i, n in self.post.get(w, ()):
                s[i] += self.idf[w] * n * (self.k1 + 1) / (
                    n + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s
