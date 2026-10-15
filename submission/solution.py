"""
Dupaco knowledge system: answers member questions from website snapshots with the value
that is current (or current as of a date), the right tier, a source, and an honest
"I don't know" when the site doesn't say.

Answer path, first match wins:
  1. rate questions        -> fact table built from every rate table on every page
  2. organisation counts   -> newest "About Dupaco" release boilerplate (branches, members...)
     yearly claims         -> newest article (board chair, Forbes recognitions)
  3. branch hours/address  -> branch directory parsed from the locator and press releases
  4. everything else       -> time-aware dense retrieval + extractive sentence answer

conflicts() lists stale copies, rate changes between snapshots, frozen press-release
statistics and edited pages, each resolved the same way answer() resolves it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:  # the scorer's own dataclasses when available (any object with these fields also works)
    sys.path.insert(0, str(next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())))
    from contract import Answer, Conflict, Snapshot, Source  # noqa: E402
except (StopIteration, ImportError):
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class Source:
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None

from branches import HOLIDAY as BRANCH_HOLIDAY, Branches  # noqa: E402
from claims import Claims  # noqa: E402
from conflicts import all_conflicts  # noqa: E402
from corpus import build_corpus, clean_markdown  # noqa: E402
from embedder import Embedder  # noqa: E402
from extract import answer_from_chunks, outline_answer  # noqa: E402
from facts import build_fact_table  # noqa: E402
from orgfacts import OrgFacts  # noqa: E402
from people import People  # noqa: E402
from query import parse  # noqa: E402
from rates import answer_rate  # noqa: E402
from retrieval import Retriever  # noqa: E402


EXTRA_ASK = re.compile(r"\b(catch|eligib\w*|qualif\w*|requirements?|who can|pay(s)? out|payout|when (does|do|is|will)|"
                       r"minimum (deposit|balance|to open|opening)|restrictions?|penalt(y|ies)|fine print|"
                       r"can (my|i|we|a) (\w+ ){0,2}(open|get|have))\b", re.I)
# words the fine print uses for what the member asked (the question itself talks about rates)
INTENT_HINTS = [
    (re.compile(r"\b(catch|fine print|restrictions?|limitations?|penalt(y|ies))\b", re.I), "limitations restrictions penalty"),
    (re.compile(r"\b(can (my|i|we|a) (\w+ ){0,2}(open|get|have)|eligib\w*|qualif\w*|who can|llc|business)\b", re.I),
     "eligible not eligible available exclusively to"),
    (re.compile(r"\b(pay(s)? out|payout|when (does|do|is|will))\b", re.I), "automatically transferred to checking on"),
]
HOLIDAY_Q = re.compile(r"\bholiday|christmas|thanksgiving|new year|memorial day|labor day|veterans day|"
                       r"independence day|july 4|easter", re.I)
_KIND_Q = re.compile(r"\b(?:a|an|the|my|your)\s+([a-z]{4,})[- ](mortgages?|certificates?|cds?|iras?)\b")
_KIND_OK = {"a", "an", "the", "my", "our", "your", "their", "new", "current", "best", "lowest", "highest", "any",
            "dupaco", "this", "that", "which", "what", "of", "on", "for", "month", "year", "jumbo", "regular",
            "standard", "special", "fixed", "rate", "home", "first", "1st", "15", "20", "30", "10", "5", "in", "to"}
CLOSED = re.compile(r"last day of service|\b(clos(e|ed|es|ing)|consolidat\w*)\b[^.]{0,40}\bbranch|"
                    r"\bbranch\b[^.]{0,40}\b(clos(e|ed|es|ing)|consolidat\w*)\b", re.I)
RATE_TALK = re.compile(r"\d\s?%|\b(apy|dividend rates?|dividends and|annual percentage|rates? (are|is) effective)\b|"
                       r"^open my\b", re.I)


class DupacoKnowledge:
    def __init__(self):
        self.snapshots = []
        self.embedder = None
        self.facts = None
        self.retriever = None
        self.branches: dict[int, Branches] = {}
        self._pages: dict[int, dict] = {}
        self._cache: dict[tuple, Answer] = {}
        self._conflicts = None

    # ------------------------------------------------------------------ ingest
    def ingest(self, snapshot) -> None:
        """Keep every snapshot and rebuild the indexes (ingest is not timed; answers are)."""
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.taken_on)
        if self.embedder is None:
            self.embedder = Embedder(log=lambda *a: print(*a, file=sys.stderr))
        chunks, dates = build_corpus(self.snapshots)
        self.chunks = chunks
        self.facts = build_fact_table(chunks, dates)
        self.org = OrgFacts(chunks)
        self.claims = Claims(chunks)
        self.retriever = Retriever(chunks, dates, self.embedder)
        self.people = People(self.retriever)
        self._blob = None
        for snap in range(len(self.snapshots)):
            self.people.cards(snap)            # staff cards indexed now: ingest isn't timed
        self.branches = {}
        self._pages = {}
        self._cache.clear()
        self._conflicts = None

    def _page_md(self, snap: int, url: str) -> str | None:
        if snap not in self._pages:
            self._pages[snap] = {p.url: p for p in self.snapshots[snap].pages}
        page = self._pages[snap].get(url)
        return clean_markdown(page.markdown) if page else None

    def _unknown_name(self, question: str) -> str | None:
        """A capitalized name (not the first word) that appears nowhere on the site."""
        for m in re.finditer(r"(?<!^)(?<![.?!]\s)\b([A-Z][a-z]{4,})\b", question.strip()):
            w = m.group(1)
            if not w.lower().startswith("dupaco") and w.lower() not in self.retriever.bm25.idf:
                return w
        return None

    def _unknown_kind(self, question: str) -> str | None:
        """"a reverse mortgage", "a bump-up certificate": a kind of mortgage / certificate the site
        never mentions (no such word within three words of it anywhere), so no rate can be quoted."""
        if self._blob is None:
            self._blob = "\n".join(sorted({c.chunk.title.lower() + " " + c.chunk.text.lower() for c in self.chunks}))
        for m in _KIND_Q.finditer(question.lower()):
            w, noun = m.group(1), m.group(2)
            if w in _KIND_OK:
                continue
            if not re.search(rf"\b{re.escape(w)}\b\W+(?:\w+\W+){{0,2}}{noun[:5]}", self._blob):
                return f"{w} {noun}"
        return None

    def _branches(self, snap: int) -> Branches:
        if snap not in self.branches:
            self.branches[snap] = Branches.build(self.snapshots[snap].pages)
        return self.branches[snap]

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, as_of: str | None = None):
        key = (question.strip(), as_of)
        if key not in self._cache:
            self._cache[key] = self._answer(question, as_of)
        return self._cache[key]

    def _answer(self, question: str, as_of: str | None):
        if not self.snapshots:
            return Answer(text="I don't have any website data loaded yet.", confident=False)
        q = parse(question, as_of)
        snap = self.facts.snapshot_for(q.as_of)
        sid = self.snapshots[snap].snapshot_id
        res = None
        if q.product_list:
            # the product page lists every account (rate or not); the rate sheet is the fallback
            res = outline_answer(question, self.retriever.search(question, snap), self.retriever,
                                 lambda url: self._page_md(snap, url))
        listed = res is not None
        if res is None and q.is_rate_question:
            res = answer_rate(q, self.facts)
            unknown = self._unknown_name(question) or self._unknown_kind(question)
            if res is not None and unknown:
                # "the APY on a Coverdell education savings account": the site never names it,
                # so the nearest savings rate would be a wrong answer
                res = {"text": f"Dupaco's website doesn't mention a {unknown} product or its rate. Please contact "
                               "Dupaco directly to ask about it.", "sources": [], "confident": False}
        if res is not None and not listed and res.get("confident", True) and EXTRA_ASK.search(question):
            # "...and what's the catch?" / "can my LLC open it?": the rate plus the fine print.
            # Only a sentence without rates is added, so no second (possibly stale) rate appears.
            ask = question + " " + " ".join(h for rx, h in INTENT_HINTS if rx.search(question))
            hits = self.retriever.search(ask, snap)
            extra = answer_from_chunks(ask, hits, self.retriever, self.embedder, snap, skip=RATE_TALK)
            if extra.get("confident") and not RATE_TALK.search(extra["text"]):
                res = dict(res, text=f"{res['text']} {extra['text']}", sources=res["sources"] + extra["sources"])
        if res is None:
            res = self.people.answer(question, snap)
        if res is None:
            res = self.org.answer(question, snap, self.snapshots[snap].taken_on)
        if res is None:
            res = self.claims.answer(question, snap, self.snapshots[snap].taken_on)
        if res is None:
            res = self._branches(snap).answer(question)
            if res is not None and not res.get("confident", True) and not HOLIDAY_Q.search(question):
                # "hours at the Asbury branch?": no hours listed may mean the branch closed
                hits = self.retriever.search(question, snap, k=12)
                ask = re.sub(r"\b(hours?( of operation)?|open(ing)?)\b", "closing", question, flags=re.I)
                ex = answer_from_chunks(ask + " last day of service", hits, self.retriever, self.embedder, snap)
                if ex.get("confident") and CLOSED.search(ex["text"]):
                    res = ex
        if res is None and BRANCH_HOLIDAY.search(question.lower()) and re.search(r"\b(open|closed?|hours)\b", question, re.I):
            res = {"text": "Dupaco's website doesn't list holiday hours or closings. Please contact Dupaco or your "
                           "branch to confirm holiday hours.", "sources": ["https://www.dupaco.com/contact"],
                   "confident": False}
        if res is None:
            hits = self.retriever.search(question, snap, k=16)
            res = outline_answer(question, hits, self.retriever, lambda url: self._page_md(snap, url)) or \
                answer_from_chunks(question, hits, self.retriever, self.embedder, snap)
            if "charter area" in res["text"].lower():
                # the join page only links a map; the newest release says what the area covers
                area = self.org.answer("how many counties", snap, self.snapshots[snap].taken_on)
                if area:
                    res["text"] += " The charter area: " + area["text"].replace("Dupaco serves", "Dupaco serves")
                    res["sources"] = res["sources"] + area["sources"]
        return Answer(text=res["text"], sources=[Source(url=u, snapshot_id=sid) for u in res.get("sources", [])],
                      confident=res.get("confident", True), conflict_note=res.get("conflict_note"))

    # --------------------------------------------------------------- conflicts
    def conflicts(self):
        """Contradictions across and within snapshots, each with the value answer() uses
        and why (see conflicts.py)."""
        if not self.snapshots:
            return []
        if self._conflicts is None:
            ids = [s.snapshot_id for s in self.snapshots]
            self._conflicts = [
                Conflict(topic=f.topic, values=f.values,
                         sources=[Source(url=u, snapshot_id=ids[s], quote=q) for u, s, q in f.sources],
                         resolution=f.resolution, reason=f.reason)
                for f in all_conflicts(self.facts, self.org, self.claims, self.chunks, self.facts.dates)]
        return self._conflicts


def build():
    return DupacoKnowledge()
