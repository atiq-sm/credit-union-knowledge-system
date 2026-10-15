"""
Branch directory and hours.

The website has no hours page: lobby / drive-up hours exist only as "**Lobby**" and
"**Drive-up**" blocks inside branch-opening press releases, and many branches have no
published hours at all. The branch list comes from the site's own branch drop-down
("Galena: Oldenburg Ln."), addresses from the /contact locator and the releases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from corpus import article_date, clean_markdown

_DAY_LINE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?(?:\s*(?:-|–|through|to|&|and)\s*[A-Z][a-z]+\.?)?\s*:\s*\S.*$")
_BLOCK = re.compile(r"^\*\*\s*(Lobby|Drive[- ]?up|Drive[- ]?thru)\s*(?:hours)?\s*:?\s*\*\*\s*:?\s*(.*)$", re.I)
_ADDRESS = re.compile(r"\b(\d{2,5}\s+(?:[NSEW]\.?\s+)?(?:[A-Z0-9][\w.']*\s+){0,4}?(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|"
                      r"Dr|Drive|Ln|Lane|Way|Hwy|Highway|Pkwy|Parkway|Ct|Court)\b\.?(?:\s+(?:NE|NW|SE|SW)\b)?)")
_PHONE = re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b")
_FEATURE = re.compile(r"\b(drive[- ]?up window|live video teller|24-hour atm|atm)\b", re.I)
FEATURE_Q = re.compile(r"\b(does|do|is there|are there|has|have)\b.*\b(drive[- ]?(up|thru|through)|video tellers?|atms?)\b", re.I)
_LOCATOR = re.compile(r"Dupaco Branch\s*\n+\s*(\d[^\n]+?,\s*([A-Za-z .]+?),\s*(IA|IL|WI),\s*\d{5})")

HOURS_INTENT = re.compile(r"\bhours?\b|\bwhat time\b|\bopen (on|until|till|tomorrow|saturdays?|sundays?)\b|\bopen at \d|"
                          r"\b(branch|branches|lobby|go|visit|open)\b.*\b(on|this) (a |the )?(saturday|sunday|weekend)s?\b|"
                          r"\b(lobby|drive[- ]?(up|thru|through))\b|\bclose[sd]? (at|on saturdays?|on sundays?)\b")
# "volunteer hours", "service hours": hours that are not opening hours
NOT_OPENING_HOURS = re.compile(r"\b(volunteer(ed|ing)?|service|community|credit|work(ed)?|billable|employee)\s+hours\b|"
                               r"\bhours (volunteered|logged|donated|served|of training|in training)\b|\bhow many hours\b|\btraining\b")
NOT_BRANCH_HOURS = re.compile(r"\b(video tellers?|itms?|call center|contact center|member care|phone|chat|customer service|"
                              r"support|financial services|insurance)\b")
HOLIDAY = re.compile(r"\bholidays?\b|\bthanksgiving\b|\bchristmas\b|\bnew year'?s?\b|\bveterans day\b|\bmemorial day\b|"
                     r"\blabor day\b|\bjuly 4|\bfourth of july\b|\bindependence day\b|\beaster\b|\bmlk\b|\bjuneteenth\b")
FEATURE_KINDS = [   # (answer label, question regex, locator feature regex)
    ("drive-up window", re.compile(r"\bdrive[- ]?(up|thru|through)"), re.compile(r"drive[- ]?up", re.I)),
    ("live video teller", re.compile(r"\bvideo tellers?\b"), re.compile(r"video teller", re.I)),
    ("24-hour ATM", re.compile(r"\batms?\b"), re.compile(r"\batm\b", re.I)),
]
FEATURE_LIST_Q = re.compile(r"\b(which|what|list|any|all)\b.*\b(branch|branches|locations|offices)\b.*\b(have|has|with|offer)")
LIST_INTENT = re.compile(r"\b(list|which|what|all|every)\b.*\b(branches|locations|offices)\b")
CLOSURE = re.compile(r"\bstill open\b|\bclos(ing|ed|ure)\b|\bshut(ting)? down\b|\bpermanently\b")
ADDRESS_INTENT = re.compile(r"\baddress\b|\bwhere is\b|\bwhere's\b|\blocated\b|\blocation of\b|\bdirections\b")


@dataclass
class Branch:
    city: str
    street: str                      # drop-down label, e.g. "Oldenburg Ln."
    address: str | None = None
    phone: str | None = None
    lobby: list[str] = field(default_factory=list)
    drive_up: list[str] = field(default_factory=list)
    source: str | None = None
    published: str | None = None
    features: list[str] = field(default_factory=list)   # locator line: "Live Video Teller · 24-hour ATM"

    @property
    def name(self) -> str:
        return f"{self.city} ({self.street})" if self.street else self.city


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("’", "'")).split().__str__()


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"st", "rd", "ave", "dr", "ln", "blvd", "hwy", "way",
                                                        "road", "street", "avenue", "drive", "lane", "e", "w",
                                                        "n", "s", "so", "the", "and"}


class Branches:
    def __init__(self):
        self.branches: list[Branch] = []
        self.cities: set[str] = set()

    @classmethod
    def build(cls, pages) -> "Branches":
        """pages: Page objects of one snapshot."""
        self = cls()
        for p in pages:                                   # 1. drop-down: the branch list
            i = p.markdown.find("Select a Branch")
            if i >= 0:
                line = p.markdown[i + len("Select a Branch"):].split("\n", 1)[0]
                for part in re.split(r"(?<=[a-z.\d])(?=[A-Z][A-Za-z ]+: )|(?<=[A-Z]{2})(?=[A-Z][a-z]+: )", line):
                    if ": " in part:
                        city, street = part.split(": ", 1)
                        self._get(city.strip(), street.strip())
                break
        for p in pages:                                   # 2. locator addresses
            if p.url.rstrip("/").endswith("/contact"):
                text = clean_markdown(p.markdown)
                for m in _LOCATOR.finditer(text):
                    b = self._match_address(m.group(2).strip(), m.group(1))
                    b.address = b.address or m.group(1).strip()
                    after = text[m.end():].split("###", 1)[0]
                    feat = next((ln for ln in after.splitlines() if _FEATURE.search(ln)), None)
                    if feat and not b.features:
                        b.features = [f.strip() for f in re.split(r"\s*[·•|]\s*", feat.strip()) if f.strip()]
        for p in pages:                                   # 3. hours blocks in releases
            if "**Lobby" not in p.markdown and "**Drive" not in p.markdown:
                continue
            text = clean_markdown(p.markdown)
            hours = _parse_hours(text)
            if not hours:
                continue
            # the branch's city: most mentioned (title counts double) in the story before the
            # hours block; case-sensitive so the "DUBUQUE, Iowa" dateline doesn't count
            story = text[:min(i for i in (text.find("**Lobby"), text.find("**Drive")) if i >= 0)]
            counts = {c: len(re.findall(rf"\b{re.escape(c)}\b", story)) + 2 * (c in p.title) for c in self.cities}
            if counts and not max(counts.values()):       # only named in an upper-case dateline
                counts = {c: len(re.findall(rf"\b{re.escape(c)}\b", story, re.I)) for c in self.cities}
            city = max(counts, key=lambda c: (counts[c], len(c))) if counts and max(counts.values()) else None
            if city is None:
                continue
            addr = _ADDRESS.search(story) or _ADDRESS.search(text)
            b = self._match_address(city, addr.group(1) if addr else "", story)
            pub = article_date(p.url, text)
            if b.published and pub and b.published > pub:
                continue                                  # keep the newest announcement
            b.lobby, b.drive_up = hours.get("lobby", []), hours.get("drive_up", [])
            b.address = b.address or (addr.group(1) if addr else None)
            ph = _PHONE.search(text)
            b.phone = ph.group(0) if ph else b.phone
            b.source, b.published = p.url, pub
        return self

    def _get(self, city: str, street: str) -> Branch:
        self.cities.add(city)
        for b in self.branches:
            if b.city == city and b.street == street:
                return b
        b = Branch(city, street)
        self.branches.append(b)
        return b

    def _match_address(self, city: str, address: str, context: str = "") -> Branch:
        """Branch in `city` whose drop-down street best matches the address/context."""
        cands = [b for b in self.branches if b.city.lower() == city.lower()]
        hay = _words(address + " " + context)
        if cands:
            best = max(cands, key=lambda b: len(_words(b.street) & hay))
            if len(cands) == 1 or _words(best.street) & hay:
                return best
        return self._get(city, "")

    # ---------------------------------------------------------------- answers
    def find(self, question: str) -> list[Branch]:
        """Branches the question names (street words beat city names)."""
        q = _words(question)
        by_street = [b for b in self.branches if b.street and (_words(b.street) - {"1st"}) & q
                     and len(_words(b.street) & q) >= 1]
        if by_street:
            return by_street
        return [b for b in self.branches if b.city.lower() in question.lower()]

    def answer(self, question: str) -> dict | None:
        low = question.lower()
        cities = [c for c in self.cities if re.search(rf"\b{re.escape(c.lower())}\b", low)]
        feat = next(((label, rx) for label, q_rx, rx in FEATURE_KINDS if q_rx.search(low)), None)
        if feat and FEATURE_LIST_Q.search(low) and not re.search(r"\b(hours?|time|open until)\b", low):
            # "which branches around Dubuque have a drive-up window?": the locator's feature lines
            have = [b for b in self.branches if b.address and any(feat[1].search(f) for f in b.features)]
            if cities and not re.search(r"\b(around|near|nearby|area|close to)\b", low):
                have = [b for b in have if b.city in cities]
            if have:
                text = f"Dupaco's branch locator lists a {feat[0]} at: " + \
                       "; ".join(f"{b.name}, {b.address}" for b in have) + "."
                return {"text": text, "sources": ["https://www.dupaco.com/contact"], "confident": True}
        if LIST_INTENT.search(low) and cities and not HOURS_INTENT.search(low):
            # the branch locator lists the open branches with their addresses; closed ones
            # (Asbury, the Hy-Vee branch) linger only in old drop-downs without an address
            listed = [b for b in self.branches if b.city in cities and b.address]
            if listed:
                text = f"Dupaco's {' and '.join(cities)} branches: " + \
                       "; ".join(f"{b.name}, {b.address}" for b in listed) + "."
                return {"text": text, "sources": ["https://www.dupaco.com/contact"], "confident": True}
        if FEATURE_Q.search(question) and not re.search(r"\b(hours?|time|open|close)\b", low):
            named = [b for b in self.find(question) if b.features]
            if len(named) == 1:
                b = named[0]
                return {"text": f"The {b.name} branch" + (f" at {b.address}" if b.address else "") +
                                f" lists: {' · '.join(b.features)}." + (f" Phone: {b.phone}." if b.phone else ""),
                        "sources": ["https://www.dupaco.com/contact"], "confident": True}
        if HOURS_INTENT.search(low) and not CLOSURE.search(low) and not NOT_OPENING_HOURS.search(low):
            if NOT_BRANCH_HOURS.search(low):
                return None                     # video tellers, phone lines: not branch hours
            if HOLIDAY.search(low):
                return {"text": "Dupaco's website doesn't list holiday hours. Please contact Dupaco or your "
                                "branch to confirm holiday hours.",
                        "sources": ["https://www.dupaco.com/contact"], "confident": False}
            return self._hours(question, low)
        if ADDRESS_INTENT.search(low):
            named = self.find(question)
            with_addr = [b for b in named if b.address]
            if with_addr:
                text = "; ".join(f"The {b.name} branch is at {b.address}" + (f" (phone {b.phone})" if b.phone else "")
                                 for b in with_addr[:3]) + "."
                return {"text": text, "sources": ["https://www.dupaco.com/contact"] +
                        [b.source for b in with_addr if b.source], "confident": True}
        return None

    def _hours(self, question: str, low: str) -> dict | None:
        drive = bool(re.search(r"drive[- ]?(up|thru|through)", low))
        named = self.find(question)
        if named:
            known = [b for b in named if (b.drive_up if drive else b.lobby)]
            if not known:
                label = named[0].name if len(named) == 1 else named[0].city
                return {"text": f"Dupaco's website doesn't list {'drive-up' if drive else 'lobby'} hours for the "
                                f"{label} branch. Please contact Dupaco or the branch directly to confirm its hours.",
                        "sources": ["https://www.dupaco.com/contact"], "confident": False}
            # "When does Waukee close on Saturday?" names neither: lobby and drive-up differ, give both;
            # "lobby hours ... and is the drive-up open Saturdays?" names both
            lobby_too = drive and bool(re.search(r"\blobby\b", low))
            both = (not drive and not re.search(r"\blobby\b", low)) or lobby_too
            parts = []
            for b in known[:2]:
                if lobby_too and b.lobby:
                    parts.append(self._schedule(b, False))
                parts.append(self._schedule(b, drive))
                if both and not lobby_too and b.drive_up:
                    parts.append(self._schedule(b, True))
            if len(known) == 1 and known[0].address:
                parts.append(f"The branch is at {known[0].address}." +   # "where is it and what are its hours?"
                             (f" Phone: {known[0].phone}." if known[0].phone else ""))
            return {"text": " ".join(parts),
                    "sources": [b.source for b in known[:2]] + ["https://www.dupaco.com/contact"], "confident": True}
        # no branch named: hours differ by branch, so give the distinct published schedules
        seen, picks = set(), []
        for b in sorted(self.branches, key=lambda b: b.published or "", reverse=True):
            sched = tuple(re.sub(r"\s*[–—-]\s*", "-", ln).rstrip(".") for ln in (b.drive_up if drive else b.lobby))
            if sched and sched not in seen:
                seen.add(sched)
                picks.append(b)
        if not picks:
            return None
        intro = ("No. Hours vary by branch. " if re.search(r"\b(same|all|every)\b", low) else
                 "Hours vary by branch. ")
        text = intro + " ".join(self._schedule(b, drive) for b in picks[:4]) + \
            " Which branch would you like hours for?"
        return {"text": text, "sources": [b.source for b in picks[:4]] + ["https://www.dupaco.com/contact"],
                "confident": True}

    @staticmethod
    def _schedule(b: Branch, drive: bool) -> str:
        lines = [ln.rstrip(";") for ln in (b.drive_up if drive else b.lobby)]
        return f"{b.name} {'drive-up' if drive else 'lobby'}: {'; '.join(lines)}" + ("" if lines[-1].endswith(".") else ".")


def _parse_hours(text: str) -> dict:
    out, current = {}, None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _BLOCK.match(s)
        if m:
            current = "lobby" if m.group(1).lower() == "lobby" else "drive_up"
            out[current] = []
            if m.group(2) and _DAY_LINE.match(m.group(2).strip()):    # "**Lobby** Monday-Thursday: ..."
                out[current].append(m.group(2).strip())
        elif current and _DAY_LINE.match(s):
            out[current].append(s)
        elif current:
            current = None
    return {k: v for k, v in out.items() if v}
