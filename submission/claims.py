"""
Claims that each year's article restates with a new value: who chairs the board, how
many times Forbes has named Dupaco a top credit union. Old articles are never edited, so
the site says "fourth time" and "fifth time", or names three different board chairs, at
once. The newest article published on or before the snapshot wins.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

_ORD = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
PATTERNS = {
    "forbes": re.compile(rf"\b({_ORD}) (?:time|year in a row|consecutive year),? Forbes\b", re.I),
    "chair": re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2}), chair of the board\b|"
                        r"\bBoard Chair(?:person)? ([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b"),
}
QUESTION = {
    "chair": re.compile(r"\bchair(man|woman|person)?\b|\bboard president\b", re.I),
    "forbes": re.compile(r"\bforbes\b", re.I),
}


@dataclass
class Claim:
    kind: str
    value: str
    url: str
    published: str
    snapshots: set
    sentences: list
    index: int                     # the claim's sentence in `sentences`

    @property
    def sentence(self) -> str:
        return self.sentences[self.index]


def sentences(text: str) -> list[str]:
    # no split after initials and "U.S."
    split = re.compile(r"(?<=[.!?])(?<![A-Z]\.[A-Z]\.)(?<!\s[A-Z]\.)\s+")
    return [s.strip() for para in text.split("\n") for s in split.split(para) if s.strip()]


_DATELINE = re.compile(r"^[A-Z][A-Z .]+, [A-Z][a-z]+\.?\s*[—–-]+\s*")


def _fmt(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{calendar.month_name[int(m)]} {int(d)}, {y}"


class Claims:
    def __init__(self, corpus_chunks):
        self.obs: dict[str, list[Claim]] = {k: [] for k in PATTERNS}
        for cc in corpus_chunks:
            if not cc.published or cc.chunk.kind == "table":
                continue
            pub = max(cc.published.values())
            sents = sentences(cc.chunk.text)
            for i, s in enumerate(sents):
                m = PATTERNS["forbes"].search(s)
                if m and re.search(r"recogni|named|rank", s):
                    self.obs["forbes"].append(Claim("forbes", m.group(1).lower(), cc.chunk.url, pub,
                                                    set(cc.snapshots), sents, i))
                for m in PATTERNS["chair"].finditer(s):
                    self.obs["chair"].append(Claim("chair", m.group(1) or m.group(2), cc.chunk.url, pub,
                                                   set(cc.snapshots), sents, i))

    def current(self, kind: str, snap: int, taken_on: str) -> Claim | None:
        cands = [c for c in self.obs[kind] if snap in c.snapshots and c.published <= taken_on]
        return max(cands, key=lambda c: (c.published, c.url)) if cands else None

    def history(self, kind: str) -> dict[str, list[Claim]]:
        out: dict[str, list[Claim]] = {}
        for c in self.obs[kind]:
            out.setdefault(c.value, []).append(c)
        return out

    def answer(self, question: str, snap: int, taken_on: str) -> dict | None:
        if QUESTION["chair"].search(question) and not re.search(r"\bcommittee\b", question, re.I):
            c = self.current("chair", snap, taken_on)
            if c:
                text = (f"{c.value} is chair of the Dupaco board of directors, according to the "
                        f"most recent announcement ({_fmt(c.published)}).")
                # "I read Ellen Goodmann Miller is chair. Still true?": her role in that announcement
                for name in re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})\b", question):
                    if name == c.value or name.split()[0] in ("Dupaco", "Is", "Who", "What", "Does", "I"):
                        continue
                    role = next((m.group(1) for s in c.sentences
                                 for m in [re.search(rf"\b{re.escape(name)}, ([a-z][a-z ]+?)(?:,|\.|$)", s)] if m), None)
                    if role:
                        text += f" {name} is {role}."
                return {"text": text, "sources": [c.url], "confident": True}
        if QUESTION["forbes"].search(question) and not re.search(r"\bemployer|\bworkplace|\bwork\b", question, re.I):
            c = self.current("forbes", snap, taken_on)
            if c:
                around = " ".join(c.sentences[max(0, c.index - 1):c.index + 2])
                return {"text": f"In its most recent announcement ({_fmt(c.published)}): " + _DATELINE.sub("", around),
                        "sources": [c.url], "confident": True}
        return None
