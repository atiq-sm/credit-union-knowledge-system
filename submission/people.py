"""
"Who runs / heads up / leads <unit>?": staff cards.

Staff listings are cards split one field per block ("Andrew Houy" / "AVP, Insurance Services" /
"800-373-7600, Ext. 2219"), each card often its own tiny chunk that dense retrieval ranks low.
The cards (extract._records) of every live page are indexed per snapshot; a lead question picks
the card whose title names the asked unit, the most senior title first (the business page's
older "Insurance Agency Manager" loses to the personal page's "AVP, Insurance Services").
"""
from __future__ import annotations

import re

from bm25 import tokenize
from extract import _records

LEAD_Q = re.compile(r"\bwho (runs|heads|leads|manages|oversees|directs|supervises)\b|\bheads? up\b|"
                    r"\bwho('s| is) (the )?(head|leader|director|manager|in charge)\b|\bin charge of\b", re.I)
TITLE = re.compile(r"\b(president|ceo|chief|svp|evp|vp|avp|vice president|director|manager|supervisor|officer)\b", re.I)
RANK = [(re.compile(r"\bceo\b|\bpresident\b(?! of)", re.I), 9), (re.compile(r"\bchief\b", re.I), 8),
        (re.compile(r"\bevp\b|executive vice president", re.I), 7),
        (re.compile(r"\bsvp\b|senior vice president", re.I), 6), (re.compile(r"\bavp\b|assistant vice president", re.I), 4),
        (re.compile(r"\bvp\b|vice president", re.I), 5), (re.compile(r"\bdirector\b", re.I), 3),
        (re.compile(r"\bmanager\b", re.I), 2), (re.compile(r"\bsupervisor\b|\bofficer\b", re.I), 1)]
_NOT_UNIT = {"who", "runs", "heads", "head", "leads", "manages", "oversees", "directs", "supervises", "up", "name",
             "title", "please", "their", "s", "charge", "team", "department", "group", "division", "leader",
             "director", "manager", "is", "the", "of", "at"}


def _rank(title: str) -> int:
    return next((r for rx, r in RANK if rx.search(title)), 0)


class People:
    def __init__(self, retriever):
        self.r = retriever
        self._cards: dict[int, list[tuple[str, str, str]]] = {}     # snap -> (name, title, card, url)

    def cards(self, snap: int):
        if snap not in self._cards:
            out, seen = [], set()
            for i in range(len(self.r.cc)):
                cc = self.r.cc[i]
                if snap not in cc.snapshots or "\n\n" not in cc.chunk.text:
                    continue
                blocks = [b.strip() for b in re.split(r"\n\s*\n", cc.chunk.text) if b.strip()]
                for rec, _pos, _lst in _records(blocks, 0):
                    fields = [f.strip() for f in rec.split("; ")]
                    if len(fields) < 3 or not TITLE.search(fields[1]) or len(fields[0].split()) > 5:
                        continue
                    key = (fields[0], fields[1], cc.chunk.url)
                    if key not in seen:
                        seen.add(key)
                        out.append((fields[0], fields[1], rec, cc.chunk.url))
            self._cards[snap] = out
        return self._cards[snap]

    def answer(self, question: str, snap: int) -> dict | None:
        if not LEAD_Q.search(question):
            return None
        unit = {w for w in tokenize(question) if w not in _NOT_UNIT and len(w) > 2}
        if not unit:
            return None
        best = None
        for name, title, rec, url in self.cards(snap):
            hit = len(unit & set(tokenize(title)))
            if hit == 0:
                continue
            key = (hit, _rank(title))
            if best is None or key > best[0]:
                best = (key, name, title, rec, url)
        if best is None or best[0][1] == 0:
            return None
        _, name, title, rec, url = best
        contact = "; ".join(f for f in rec.split("; ")[2:] if not f.startswith("["))
        text = f"{name}, {title}" + (f" ({contact})" if contact else "") + "."
        return {"text": text, "sources": [url], "confident": True}
