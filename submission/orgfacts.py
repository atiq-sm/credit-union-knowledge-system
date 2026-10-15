"""
Organisation statistics that press releases restate in their "About Dupaco" boilerplate,
each frozen at its publication date: branch count, members, employees, counties, assets.
Old releases are never edited, so the site carries 18, 19, 20, 22, 23 and 24 "branch
offices" at once. The newest release published on or before the snapshot wins.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

STATS = {   # name -> (pattern with group 1 = value incl. its qualifier, answer template)
    "branches": (re.compile(r"\b(\d{1,2})\s+(?:full-service\s+)?branch(?:es| offices?| office locations| locations)\b", re.I),
                 "Dupaco has {v} branch office locations"),
    "members": (re.compile(r"(?:membership has grown to|serves|serving)\s+((?:more than|over|nearly)\s+[\d,]{6,})", re.I),
                "Dupaco has {v} members"),
    "employees": (re.compile(r"\b(?:has|employs)\s+((?:more than|over|nearly)\s+\d{3,4})\s+employees", re.I),
                  "Dupaco has {v} employees"),
    "counties": (re.compile(r"\bresidents in\s+(\d{2,3})\s+counties\b", re.I),
                 "Dupaco serves residents in {v} counties throughout Iowa, northwest Illinois and southwest Wisconsin"),
    "assets": (re.compile(r"\bassets\s+((?:of\s+)?(?:exceeding|of more than|of nearly|nearly|more than|over)\s+\$[\d.]+\s+billion)", re.I),
               "Dupaco has assets {v}"),
}
QUESTION = {
    "branches": re.compile(r"\bhow many (branch(es)?|locations|branch offices|offices)\b|(?<!phone )(?<!fax )\bnumber of\b.*\b(branch|location|office)", re.I),
    "members": re.compile(r"\bhow many (members|member-owners|people (are|belong))\b(?! (are |have )?(using|use|used|who|that|enrolled|signed|in \w+ (program|tool)))|\bnumber of members\b|\bmembership (size|count)\b", re.I),
    "employees": re.compile(r"\bhow many (employees|people|staff)\b(?!.*\b(used|took|take|received|volunteer|attended|vto))"
                            r"(?=.*\b(have|employ|work)\b)", re.I),
    "counties": re.compile(r"\bhow many counties\b|\bcounties\b.*\bserve\b", re.I),
    "assets": re.compile(r"\b(total )?assets\b|\bhow big is\b", re.I),
}


# the Owner's Manual newsletter prints a dated "FINANCIAL STRENGTH" box (newer than any release)
_AS_OF = re.compile(r"\bAs of:?\s*(January|February|March|April|May|June|July|August|September|October|"
                    r"November|December)\s+(\d{1,2}),\s+(20\d\d)")
_STATEMENT_MEMBERS = re.compile(r"\bMembers:\s*([\d,]{6,})")


@dataclass
class Stat:
    name: str
    value: str
    url: str
    published: str
    snapshots: set
    sentence: str


def _sentence(text: str, start: int, end: int) -> str:
    a = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    b = text.find(". ", end)
    return text[a + 1:(b + 1) if b > 0 else len(text)].strip()


def _fmt(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{calendar.month_name[int(m)]} {int(d)}, {y}"


class OrgFacts:
    def __init__(self, corpus_chunks):
        self.obs: dict[str, list[Stat]] = {k: [] for k in STATS}
        for cc in corpus_chunks:
            m, n = _AS_OF.search(cc.chunk.text), _STATEMENT_MEMBERS.search(cc.chunk.text)
            if m and n:
                iso = f"{m.group(3)}-{list(calendar.month_name).index(m.group(1)):02d}-{int(m.group(2)):02d}"
                self.obs["members"].append(Stat("members", n.group(1), cc.chunk.url, iso, set(cc.snapshots),
                                                f"Members: {n.group(1)} (as of {m.group(0)[6:].strip(': ')})"))
            if not cc.published:
                continue
            pub = max(cc.published.values())
            for name, (rx, _) in STATS.items():
                for m in rx.finditer(cc.chunk.text):
                    self.obs[name].append(Stat(name, m.group(1), cc.chunk.url, pub, set(cc.snapshots),
                                               _sentence(cc.chunk.text, m.start(), m.end())))

    def current(self, name: str, snap: int, taken_on: str, releases_only: bool = False) -> Stat | None:
        cands = [s for s in self.obs[name] if snap in s.snapshots and s.published <= taken_on
                 and not (releases_only and s.url.lower().endswith(".pdf"))]
        return max(cands, key=lambda s: (s.published, s.url)) if cands else None

    def answer(self, question: str, snap: int, taken_on: str) -> dict | None:
        for name, rx in QUESTION.items():
            if rx.search(question):
                st = self.current(name, snap, taken_on)
                if st is None:
                    return None
                if not st.url.lower().endswith(".pdf"):
                    return {"text": STATS[name][1].format(v=st.value)
                            + f", according to its most recent release ({_fmt(st.published)}).",
                            "sources": [st.url], "confident": True}
                text = STATS[name][1].format(v=st.value) + f" as of {_fmt(st.published)} (Owner's Manual newsletter)."
                rel = self.current(name, snap, taken_on, releases_only=True)
                if rel:
                    text += f" Its most recent press release ({_fmt(rel.published)}) said {rel.value}."
                return {"text": text, "sources": [st.url] + ([rel.url] if rel else []), "confident": True}
        return None

    def history(self, name: str) -> dict[str, list[Stat]]:
        out: dict[str, list[Stat]] = {}
        for s in self.obs[name]:
            out.setdefault(s.value, []).append(s)
        return out
