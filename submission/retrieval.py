"""
Time-aware chunk retrieval.

Dense EmbeddingGemma similarity over the chunks live in the requested snapshot (BM25
stands in for chunks without a vector, or entirely if the model is unavailable), plus
two small priors that only break near-ties:
  * page authority: rate sheet / product pages over landing pages, calculators, posts;
  * article recency: a 2026 press release over a 2021 one saying the opposite
    ("22 branch offices" vs "24 branch offices"). Undated product pages count as current.
Expired-offer pages are pushed down.
"""
from __future__ import annotations

import math
import re
from datetime import date

import numpy as np

from bm25 import BM25, tokenize
from embedder import doc_input
from facts import authority

W_AUTH = 0.03
W_RECENT = 0.04
RECENCY_YEARS = 2.0
EXPIRED_PENALTY = 0.15
_TOLL_FREE = re.compile(r"\b8(?:00|33|44|55|66|77|88)-\d{3}-\d{4}\b")
_EXPIRED = re.compile(r"offer has (expired|ended)|this offer has expired", re.I)


def _years_between(a: str, b: str) -> float:
    try:
        return (date.fromisoformat(b) - date.fromisoformat(a)).days / 365.25
    except ValueError:
        return 0.0


class Retriever:
    def __init__(self, corpus_chunks, dates: list[str], embedder):
        self.cc = corpus_chunks
        self.dates = dates
        self.emb = embedder
        self.bm25 = BM25([c.chunk.embed_text for c in corpus_chunks])
        if embedder is not None and embedder.available:
            self.D, self.have = embedder.embed_docs([doc_input(c.chunk) for c in corpus_chunks])
        else:
            self.D, self.have = None, np.zeros(len(corpus_chunks), bool)
        self.masks = [np.array([s in c.snapshots for c in corpus_chunks]) for s in range(len(dates))]
        self.auth = np.array([authority(c.chunk.url) for c in corpus_chunks], np.float32)
        self.expired = np.array([bool(_EXPIRED.search(c.chunk.title + " " + c.chunk.text)) for c in corpus_chunks])
        self._recency: dict[int, np.ndarray] = {}

    def recency(self, snap: int) -> np.ndarray:
        if snap not in self._recency:
            today = self.dates[snap]
            r = np.empty(len(self.cc), np.float32)
            for i, c in enumerate(self.cc):
                pub = c.published.get(snap) or (max(c.published.values()) if c.published else None)
                r[i] = 1.0 if pub is None else math.exp(-max(0.0, _years_between(pub, today)) / RECENCY_YEARS)
            self._recency[snap] = r
        return self._recency[snap]

    def search(self, question: str, snap: int, k: int = 8, qvec=None):
        """[(chunk index, score)] best first, restricted to chunks live in `snap`."""
        mask = self.masks[snap]
        bm = self.bm25.scores(question)
        bm = bm / (bm.max() or 1.0)
        if self.D is not None:
            qv = qvec if qvec is not None else self.emb.embed_query(question)
            dense = self.D @ qv
            dense = np.where(self.have, dense, 0.6 * bm)      # no vector: keyword score stands in
        else:
            dense = bm
        score = dense + W_AUTH * self.auth + W_RECENT * self.recency(snap) - EXPIRED_PENALTY * self.expired
        score = np.where(mask, score, -np.inf)
        k = min(k, int(mask.sum()))
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top], kind="stable")]
        return [(int(i), float(score[i]), float(dense[i])) for i in top]

    def _page(self, url: str) -> list[int]:
        if not hasattr(self, "_by_url"):
            self._by_url: dict[str, list[int]] = {}
            for i, c in enumerate(self.cc):
                self._by_url.setdefault(c.chunk.url, []).append(i)
        return self._by_url.get(url, [])

    def on_page(self, text: str, url: str, snap: int) -> bool:
        """`text` appears in some chunk of page `url` live in `snap`."""
        return any(self.masks[snap][i] and text in self.cc[i].chunk.text for i in self._page(url))

    def main_phone(self, snap: int) -> str | None:
        """The toll-free number printed on the most pages live in `snap` (the general contact line)."""
        if not hasattr(self, "_phones"):
            self._phones: dict[int, str | None] = {}
        if snap not in self._phones:
            pages: dict[str, set] = {}
            for i, c in enumerate(self.cc):
                if self.masks[snap][i]:
                    for num in _TOLL_FREE.findall(c.chunk.text):
                        pages.setdefault(num, set()).add(c.chunk.url)
            self._phones[snap] = max(pages, key=lambda n: (len(pages[n]), n)) if pages else None
        return self._phones[snap]

    def page_hits(self, ci: int, snap: int, exclude, k: int = 4, qvec=None):
        """The other chunks of chunk `ci`'s page live in `snap`, best first, scored like search()
        results: a release's list of winners sits a few chunks below its lead."""
        out = []
        for i in self._page(self.cc[ci].chunk.url):
            if i in exclude or not self.masks[snap][i]:
                continue
            d = float(self.D[i] @ qvec) if self.D is not None and qvec is not None and self.have[i] else 0.0
            s = d + W_AUTH * self.auth[i] + W_RECENT * self.recency(snap)[i] - EXPIRED_PENALTY * self.expired[i]
            out.append((i, float(s), d))
        return sorted(out, key=lambda h: -h[1])[:k]

    def keyword_hits(self, question: str, snap: int, exclude, k: int = 2, qvec=None):
        """Top BM25 chunks live in `snap` (outside `exclude`), scored like search() results.
        Rare exact terms ("Analysis checking", a person's name) that the dense model ranks low."""
        words = [w for w in set(tokenize(question)) if w in self.bm25.idf]
        if not words:
            return []
        rare = max(words, key=lambda w: self.bm25.idf[w])     # the question's most specific word ...
        has = {i for i, _ in self.bm25.post[rare]}             # ... must be in the chunk
        bm = np.where(self.masks[snap], self.bm25.scores(question), -1.0)
        out = []
        for i in np.argsort(-bm)[:k + len(exclude)]:
            if bm[i] <= 0 or int(i) in exclude or int(i) not in has:
                continue
            d = float(self.D[i] @ qvec) if self.D is not None and qvec is not None and self.have[i] else 0.0
            s = d + W_AUTH * self.auth[i] + W_RECENT * self.recency(snap)[i] - EXPIRED_PENALTY * self.expired[i]
            out.append((int(i), float(s), d))
            if len(out) == k:
                break
        return out
