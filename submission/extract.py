"""
Extractive answers for non-rate questions: pick the sentence(s) from the top retrieved
chunks that best answer the question.

Short answers on purpose: every extra sentence is a chance to quote a stale number
(an old branch count, a superseded rate), which the grader scores as a wrong answer.
"""
from __future__ import annotations

import re

import numpy as np

from bm25 import _STOP, tokenize
from corpus import strip_links

CONTACT_LINE = "You can also call Dupaco at {phone}."   # the toll-free number the site prints most
REFUSAL = ("I couldn't find that on Dupaco's website. Please contact Dupaco directly "
           "so a team member can help.")
MIN_DENSE = 0.40          # below this, nothing retrieved is about the question
UNSURE_DENSE = 0.50       # below this, the answer must also cover the question's key words
GENERIC = {"today", "now", "current", "currently", "tell", "know", "need", "want", "get", "much", "many",
           "please", "like", "would", "could", "also", "any", "there", "here", "if", "amount"}
NEAR_TIE_TOTAL = 0.10     # ... wider for "how many / how much": totals change, newest release wins
RECENT_Q = re.compile(r"\b(yet|still|anymore|any more|newest|latest|most recent(ly)?|these days|nowadays)\b", re.I)
NEAR_TIE = 0.05           # sentence scores this close count as a tie (newest article wins)
SECOND_GAP = 0.06         # a second sentence must score within this of the best
RECORD_LINE = 80          # a block this short and on one line is a card field, not prose
KEYWORD_PENALTY = 0.0
KEYWORD_EXTRA = 2         # BM25 chunks added to the sentence pool (exact rare terms)
OLD_ARTICLE_PENALTY = 0.12   # article published before the year a question asks about
PART2_SLACK = 0.15        # ... or within this when the answer so far lacks the value part 2 asks for
PART2_GAIN = 0.03        # a two-part question's second half gets its own sentence when one fits it this much better
DEBUG = False
POOL_TOP = 8              # top retrieved chunks whose sentences are candidates
PAGE_EXTRA = 0            # other chunks of the top chunk's page added to the pool
RERANK = None             # dev pilot hook: f(question, sentences) -> scores; None when shipped
RERANK_TOP = 12
RERANK_WEIGHT = 0.0

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"($])")
_ABBREV = re.compile(r"\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|dr|st|ave|rd|blvd|ln|no|inc|mr|mrs|ms|"
                     r"jr|sr|vs|etc|e\.g|i\.e|a\.m|p\.m|u\.s|ext|co|corp|hwy|(?-i:[A-Z]))\.$", re.I)   # "Robert W."
SYNONYMS = {"close": {"last", "closing", "closed"}, "member": {"join", "membership"},
            "become": {"join"}, "deadline": {"period", "apply", "application"},
            "fee": {"charge", "cost"}, "apply": {"application"}, "grant": {"gift", "award", "fund"}, "maximum": {"up", "max"}, "activate": {"activation", "activating"},
            "call": {"contact", "phone"}, "phone": {"contact", "call"}, "extension": {"ext"}}
_DOC_EXT = re.compile(r"\.(jpe?g|png|gif|svg|webp|pdf|xlsx?|csv|docx?)(\?|$)", re.I)
_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"https?://[^\s)\]>]+")

QTYPES = [   # (question regex, sentence regex that a good answer usually contains)
    ("count", re.compile(r"\bhow many\b|\bnumber of\b|\b(what|the) (was|is|were) the total\b|\btotal (number|count) of\b"), re.compile(r"\b\d[\d,]*\b")),
    ("money", re.compile(r"\bhow much\b|\bfee\b|\bcost\b|\bprice\b|\b(max(imum)?|min(imum)?|most|largest|biggest) (amount|grant|gift|award|loan|bonus)\b"), re.compile(r"\$\s?\d|\bfree\b|\bno (fee|cost|charge)\b", re.I)),
    ("date", re.compile(r"\bwhen\b|\bwhat (date|day|year)\b|\bdeadline\b|\bstill open\b"),
     re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|\b(19|20)\d\d\b", re.I)),
    ("phone", re.compile(r"\bphone\b|\bcall\b|\btelephone\b|\bextension\b|\bhow (do|can) i reach\b"), re.compile(r"\d{3}[-.)]\s?\d{3}-\d{4}")),
    ("routing", re.compile(r"\brouting\b|\baba\b"), re.compile(r"\b\d{9}\b")),
    ("person", re.compile(r"\bwho\b"), re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")),
    ("where", re.compile(r"\bwhere\b"), re.compile(r"\b(branch|street|st\.|ave|road|rd\.|located|at)\b", re.I)),
    ("domain", re.compile(r"\b(which|what) (domains?|sites|websites|urls|addresses)\b|\b(whitelist|allowlist|allow ?list)\b"),
     re.compile(r"\b[\w-]+\.(com|net|org|io)\b", re.I)),
    ("duration", re.compile(r"\bhow long\b"), re.compile(r"\b\d+\s*(business |calendar )?(days?|weeks?|months?|years?)\b", re.I)),
    ("eligible", re.compile(r"\bcan (i|we|my \w+) (still )?(get|qualify|receive|earn)\b|\beligib|\bqualif"),
     re.compile(r"\b(in)?eligib|\bqualif|\bnot (available|apply)\b|\bonly\b|\bmust\b", re.I)),
]
# questions whose answer must contain a certain kind of value. When no well-ranked sentence
# has one, the site doesn't say ("What's the daily mobile deposit limit?" must not be
# answered with how mobile deposit works), so the answer is an honest "I don't know".
_NUMWORDS = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozens?|hundreds?|thousands?|" \
            r"millions?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
NEEDS = [
    (re.compile(r"\b(salary|salaries|wages?|hourly|starting pay|pay rate|compensation|paid per hour)\b", re.I),
     re.compile(r"\$\s?\d")),
    (re.compile(r"\b(what('s| is| are)|how (much|high|big)) (the |my |a |your )?(\w+ ){0,4}(limits?|cap)\b|"
                r"\b(daily|weekly|monthly|maximum) (\w+ ){0,3}limits?\b|\bis there a (cap|limit)\b", re.I), re.compile(r"\d")),
    (re.compile(r"\b(percentage|percent)\b|\bwhat %", re.I), re.compile(r"\d\s?%|\bpercent\b", re.I)),
    (re.compile(r"\bwhat time\b|\bcut-?off time\b|\b(open|opening|business|branch|lobby|drive\S*|teller|holiday|weekend|"
                r"saturday|sunday)\w* hours\b|\bhours (of operation|are)\b|\bwhat are (the|your|their) hours\b|"
                r"\bhours (is|are) (the|it|they)\b", re.I),
     re.compile(r"\b\d{1,2}(:\d\d)?\s*([ap]\.?m\b|noon)|\bnoon\b|\b24/7\b|\b24 hours\b|\bmidnight\b", re.I)),
    (re.compile(r"\b(swift|bic)\b", re.I), re.compile(r"\b(swift|bic)\b", re.I)),
    (re.compile(r"\bhow many\b", re.I), re.compile(r"\d|" + _NUMWORDS, re.I)),
    (re.compile(r"\b(fees?|cost|price|charges?)\b|\bhow much\b", re.I),
     re.compile(r"\$\s?\d|\d\s?%|\bfree\b|\bno (\w+ )?(fees?|costs?|charges?)\b|\bwaive|\bno cost\b|\b0\b", re.I)),
]
NEEDS.append((re.compile(r"\bwhen (can|do|does|is|are|will|did|was)\b.*\b(apply|applications?|opens?|opened|due|deadline|"
                         r"start|begins?|ends?|close[sd]?|announced?|held|take place)\b", re.I), QTYPES[2][2]))
# "was there an intro rate?": a rate is "1.99%" / "3% APY", not the "85% loan-to-value" beside it
RATE_NEED = (re.compile(r"\b(intro(ductory)? |interest |dividend |promo(tional)? )?(?<!1st )(?<!first )(rates?|aprs?|apys?)\b(?! (of )?(return|change))", re.I),
             re.compile(r"\d\.\d+\s?%|\d\s?%\s*(apr|apy|dividend|intro|rate)", re.I))
NEEDS.append(RATE_NEED)
NEEDS.append((re.compile(r"\b(which|what) (domains?|sites|websites|urls)\b|\b(whitelist|allowlist)\b", re.I),
              re.compile(r"\b[\w-]+\.(com|net|org|io)\b", re.I)))
# "back at the start of June 2026": as-of wording, not something the answer must mention
AS_OF_WORDS = {"back", "start", "beginning", "end", "early", "late", "mid", "middle", "as", "january", "february",
               "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
NEED_GAP = 0.15          # a sentence with the needed value may rank this far below the best
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_ORDINAL = re.compile(r"\b\d+(st|nd|rd|th)\b", re.I)
LINK_Q = re.compile(r"\b(link|url|website|web ?page|online|sign up|apply|log ?in)\b")
# "When can nonprofits apply?" asks for a date, "Can I turn off ... online banking?" for a yes/no
NOT_LINK_Q = re.compile(r"^\s*(when|who|why|can (i|you|we) (turn|change|use|still|get)|"
                        r"how (much|many|long)|what (time|date|day))\b|"
                        r"\bcan (i|you|we) (turn|switch|shut|stop|disable|remove|opt)\b|\bwhen (does|do|is|will|can)\b|"
                        r"\b(which|what) (domains?|sites|websites|urls|addresses)\b|"
                        r"\b(deadline|due date)\b|\bwhat (does|will) (it|that|this) cost\b|\bhow much\b", re.I)
HOWTO_Q = re.compile(r"^\s*(how (do|can|should) i|how to|can i|what'?s the best way)\b", re.I)
LIST_Q = re.compile(r"\b(who (is|are) eligible|who can (join|apply|open)|eligib\w*|requirements?|qualif\w*|"
                    r"what do i need|benefits|what are the (steps|options|ways|perks))\b", re.I)
TOTAL_Q = re.compile(r"\b(total|in all|altogether|combined|overall|so far|this year|last year|in (19|20)\d\d)\b", re.I)
STREET = re.compile(r"\b\d{2,5}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][\w.]*\s+){1,3}(?:St|Street|Rd|Road|Ave|Avenue|Way|Hwy|Highway|"
                    r"Dr|Drive|Blvd|Ln|Lane|Ct|Court|Pkwy|Parkway|Circle|Pl|Place)\b")
LIST_Q_MORE = re.compile(r"\b(what (can|could) (i|you|we) do|what (services|transactions|types|kinds|fees)|"
                         r"which (services|transactions))\b", re.I)
ADDRESS_BONUS = 0.08      # "where is X": a street address answers it better than "located in Waukee"
PDF_PENALTY = 0.0         # PDF text (forms, terms) only answers when no web page says it as well
LIST_RANK = 1             # a list in one of the top LIST_RANK + 1 passages answers a list question
FOOTNOTE_FOLLOW = True    # an answer ending in a footnote mark gets the footnote
YEAR_MISMATCH = 0.08       # a sentence dated to another year than the one asked about
PAST_DEADLINE = 0.25      # a deadline already passed, for "when can I apply"
TOTAL_WORDS_ONLY = True  # a money question is a year-total question only when it says so
LIST_MORE = False         # "what can I do at ...", "what fees ..." are list questions too
FULL_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(19|20)\d\d\b", re.I)
UPCOMING_Q = re.compile(r"\b(deadline|due|next|upcoming|apply|application|open(s)? for applications)\b", re.I)
_MON_NUM = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_PARTS = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+((?:19|20)\d\d)\b", re.I)


FAMILY_GUARD = True       # a question about one loan family is not answered by another
FAMILIES = {"auto": re.compile(r"\b(cars?|auto|autos|vehicles?|motorcycles?|trucks?|suvs?)\b", re.I),
            "home": re.compile(r"\b(home loans?|mortgages?|home ?buyers?|house)\b", re.I)}


def _families(text: str) -> set[str]:
    return {f for f, rx in FAMILIES.items() if rx.search(text)}


def _iso_dates(s: str) -> list[str]:
    return [f"{m.group(3)}-{_MON_NUM[m.group(1).lower()]:02d}-{int(m.group(2)):02d}" for m in _DATE_PARTS.finditer(s)]


PLACEHOLDER = re.compile(r"\bext\.?\s*0{3,}\b|\b0{3}[-.]0{3}[-.]0{4}\b|\bx{3,}\b", re.I)   # "Ext. 0000" left in a template
CALL_TO_ACTION = re.compile(r"^(for more information|to learn more|learn more|visit|click|call us|contact us)\b", re.I)
DEFINE_Q = re.compile(r"^\s*(what is|what's|what are|tell me about|explain|describe)\b"
                      r"(?!.*\b(rate|fee|number|phone|address|url|link|hours|apy|apr|cost|price)s?\b)", re.I)


def _sentences(text: str) -> list[tuple[str, int, bool]]:
    """(sentence, position, is_list) triples. A short heading followed by bullet items also yields one
    combined candidate ("Eligible if: A; B; C"), since lists answer "who can / what are"."""
    out, pos = [], 0
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.startswith("|"):
            rows = [r for r in block.splitlines() if r.strip().startswith("|") and not re.match(r"^\|\s*:?-", r.strip())]
            for r in rows[1:]:               # fee schedules etc.: "Incoming wire transfer: $10.00"
                cells = [c.strip() for c in r.strip().strip("|").split("|") if c.strip() and c.strip() != "Col"]
                if len(cells) >= 2:
                    out.append((f"{cells[0]}: " + "; ".join(cells[1:]), pos, False))
                    pos += 1
            i += 1
            continue
        bullets, j = [], i + 1
        while j < len(blocks) and re.match(r"^[-*•]\s", blocks[j]):
            bullets += [ln.strip().lstrip("-*• ").strip() for ln in blocks[j].splitlines() if ln.strip()]
            j += 1
        if len(bullets) >= 2 and (len(block) < 120 or block.endswith(":")) and "\n" not in block:
            out.append((f"{block.rstrip(':')}: " + "; ".join(b.rstrip(".;") for b in bullets[:8]) + ".", pos, True))
        lines: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            # PDFs wrap mid-sentence ("...is $50.00 with the" / "maximum being $500.00.")
            if lines and line[:1].islower() and not re.search(r"[.!?:;]$", lines[-1]) \
                    and not re.match(r"^[-*•]\s", line):
                lines[-1] = f"{lines[-1]} {line}"
            else:
                lines.append(line)
        for line in lines:
            line = line.strip().lstrip("-*• ").strip()
            if line:
                for s in _split(line):
                    out.append((s, pos, False))
                    pos += 1
        i += 1
    return out + _records(blocks, pos)


def sentence_inputs(chunk) -> list[str]:
    """Model inputs of every candidate sentence of a chunk, exactly as answer_from_chunks embeds
    them (dev/build_sent_cache.py ships their vectors so answers rarely embed anything)."""
    from embedder import DOC_PREFIX
    out = []
    for s, _pos, _lst in _sentences(chunk.text):
        plain = _FOOTNOTE.sub("", strip_links(s))
        if len(plain) >= 25 or re.search(r"\d", plain):
            out.append(DOC_PREFIX.format(title=chunk.title or "none") + plain)
    return out


def _short(block: str) -> bool:
    return "\n" not in block and len(block) <= RECORD_LINE and not block.startswith("|") \
        and not re.match(r"^[-*•]\s", block)


def _records(blocks: list[str], pos: int) -> list[tuple[str, int, bool]]:
    """Cards and tiles split one field per block ("Randy Butler" / "Financial Consultant" /
    "800-373-7600, Ext. 2917"; "6,136" / "Volunteer Hours"). A lone field barely matches a
    question, so each short block that holds a value also yields a candidate joined with the
    up to two short blocks before it (its label), and a bare number the block after it."""
    out = []
    for k, b in enumerate(blocks):
        if not _short(b) or not re.search(r"\d|@", b):
            continue
        lo = k
        while lo > max(0, k - 2) and _short(blocks[lo - 1]) and not re.search(r"\d|@", blocks[lo - 1]):
            lo -= 1
        hi = k + 1
        if len(re.findall(r"[A-Za-z]{2,}", b)) <= 1 and hi < len(blocks) and _short(blocks[hi]):
            hi += 1                      # "6,136" is labelled by the block after it ...
            if hi < len(blocks) and len(blocks[hi]) <= 250 and "\n" not in blocks[hi] and \
                    not re.search(r"^[-*•|]", blocks[hi]):
                hi += 1                  # ... and described by the one after that
        if hi - lo < 2:
            continue
        rec = "; ".join(re.sub(r"\*\*|__", "", x).strip() for x in blocks[lo:hi])
        out.append((rec, pos, False))
        pos += 1
    return out


def _split(line: str) -> list[str]:
    """Sentence split that keeps 'Jan. 5' / 'St. Louis' / '8 a.m. to' together."""
    out = []
    for part in _SENT.split(line):
        part = part.strip()
        if not part:
            continue
        if out and _ABBREV.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(s: str) -> set[str]:
    return {_stem(w) for w in tokenize(s)}


def _qwords(q: str) -> set[str]:
    ws = set(tokenize(q))
    for w in list(ws):
        ws |= SYNONYMS.get(w, set())
    return {_stem(w) for w in ws}


def _coverage(question: str, answer_words: set[str]) -> float:
    """Share of the question's key words (each counted once) found in the answer,
    a word counting as found when it or one of its synonyms is there."""
    key = [w for w in tokenize(question) if w not in GENERIC and w not in {"about", "call?"}
           and w not in AS_OF_WORDS and not w.isdigit()]
    if not key:
        return 1.0
    found = sum(1 for w in key if ({_stem(w)} | {_stem(x) for x in SYNONYMS.get(w, ())}) & answer_words)
    return found / len(key)


def _is_lead(chunk) -> bool:
    """Chunk opens with the page title heading, i.e. it is the start of the article/page."""
    first = chunk.text.split("\n", 1)[0].strip().lower()
    return bool(chunk.title) and first[:40] == chunk.title.strip().lower()[:40]


_REL_YEAR = re.compile(r"\b(last|this|past|previous|current) (calendar |fiscal )?year\b", re.I)


def _own_year(s: str, question: str) -> bool:
    """The sentence dates its own figure ("294 used VTO in 2023"); "since 2016" from the question doesn't count."""
    return bool(set(_YEAR.findall(s)) - set(_YEAR.findall(question)))


def _relative_years(question: str, today: str) -> set[str]:
    """'last year' / 'this year' as of the snapshot date."""
    out = set()
    for m in _REL_YEAR.finditer(question):
        y = int(today[:4]) - (1 if m.group(1).lower() in ("last", "past", "previous") else 0)
        out.add(str(y))
    return out


def _published(retriever, ci: int, snap: int) -> str:
    """Article date for near-tie breaking; undated product pages count as current."""
    cc = retriever.cc[ci]
    return cc.published.get(snap) or (max(cc.published.values()) if cc.published else "9999")


BUSINESS_Q = re.compile(r"\b(business|llc|corporation)\b", re.I)
_BUSINESS_URL = re.compile(r"/business|business-")
SCOPE_PENALTY = 0.06
_WORD = re.compile(r"[a-z0-9]+")


FEE_Q = re.compile(r"\b(fees?|charges?)\b", re.I)
FEE_WORD = re.compile(r"\b(fees?|charges?|costs?|waive\w*|free|surcharge\w*|pricing)\b", re.I)
_ATTR_HEAD = r"(fees?|charges?|limits?|minimums?|deadlines?|extensions?|penalt(?:y|ies))"
ATTR_BONUS = 0.08


def _attr_phrase(question: str):
    """'What's the late payment fee on...' -> a regex for 'late payment fee(s)': the one or
    two words naming which fee/limit/deadline is asked (None when the head stands alone)."""
    words = _WORD.findall(question.lower().replace("’", "'"))
    for i, w in enumerate(words):
        if re.fullmatch(_ATTR_HEAD, w) and i:
            mods = []
            for m in reversed(words[max(0, i - 2):i]):
                if m in _STOP or m in GENERIC or m in {"the", "a", "an", "s", "what", "your"}:
                    break
                mods.insert(0, m)
            if mods:
                head = re.sub(r"(y|ies)$", "", w) if re.search(r"(y|ies)$", w) else re.sub(r"s$", "", w)
                return re.compile(r"\b" + " ".join(mods) + rf" {head}(s|y|ies)?\b")
    return None


def _fee_mods(question: str) -> set[str]:
    """{'advisory'} for 'What advisory fee ...', {'late'} for 'the late payment fee', {'foreign'} for
    'a foreign transaction fee': the leftmost qualifier is the one that names which fee."""
    words = _WORD.findall(question.lower().replace("’", "'"))
    for i, w in enumerate(words):
        if w in ("charge", "charges") and words[i + 1:i + 2] == ["for"]:
            # "what does Dupaco charge for merchant card processing": the thing charged for
            nxt = [m for m in words[i + 2:i + 5] if m not in _STOP and m not in GENERIC and m not in {"a", "an", "the"}]
            if nxt:
                return {nxt[0]}
        if w in ("fee", "fees", "charge", "charges") and i:
            mods = []
            for m in reversed(words[max(0, i - 2):i]):
                if m in _STOP or m in GENERIC or m in {"the", "a", "an", "s", "what", "your", "any", "much"}:
                    break
                mods.append(m)
            if mods:                       # "does Dupaco charge a foreign transaction fee": the fee, not "charge"
                return {mods[-1]}
    return set()


def _phrase_hits(question: str, hits, retriever, extra: int = 2) -> list:
    """Lower-ranked chunks that contain a three-word phrase of the question verbatim
    ("late payment fee" in a card's fine print), so a fact buried in a long footnote can
    still be the answer. At most `extra` chunks, most phrases first."""
    words = _WORD.findall(question.lower().replace("’", "'"))
    phrases = {" ".join(words[i:i + 3]) for i in range(len(words) - 2)
               if not any(w in _STOP or w in GENERIC for w in words[i:i + 3])}
    if not phrases:
        return []
    found = []
    for rank, h in enumerate(hits[POOL_TOP:], POOL_TOP):
        text = " ".join(_WORD.findall(retriever.cc[h[0]].chunk.text.lower()))
        n = sum(p in text for p in phrases)
        if n:
            found.append((-n, rank, h))
    return [h for *_, h in sorted(found)[:extra]]


def answer_from_chunks(question: str, hits, retriever, embedder, snap_idx: int, skip=None) -> dict:
    """hits: [(chunk index, score, dense)] from Retriever.search. `skip`: a regex of
    sentences that must not be the answer (rates, when only the fine print is wanted)."""
    if not hits or hits[0][2] < MIN_DENSE:
        return {"text": REFUSAL, "sources": [], "confident": False}
    qwords = _qwords(question)
    part2 = _second_part(question)
    part1 = question[:question.rfind(part2)] if part2 else question
    part1 = re.sub(r"\bin charge\b", " ", part1, flags=re.I)   # "who is in charge of risk" is no fee question
    qtype = next(((name, ok) for name, rx, ok in QTYPES if rx.search(part1.lower())), None)
    define = bool(DEFINE_Q.search(question))

    if LINK_Q.search(question.lower()) and not NOT_LINK_Q.search(question):
        link = _best_link(question, hits[:3], retriever)
        if link:
            return {"text": f"You can do that at {link}.", "sources": [retriever.cc[hits[0][0]].chunk.url, link],
                    "confident": True}

    qv = embedder.embed_query(question) if embedder is not None and embedder.available else None
    pool = hits[:POOL_TOP] + _phrase_hits(question, hits, retriever)
    if PAGE_EXTRA:
        pool += retriever.page_hits(hits[0][0], snap_idx, {h[0] for h in pool}, k=PAGE_EXTRA, qvec=qv)
    n_base = len(pool)                 # later additions are keyword-only chunks
    if KEYWORD_EXTRA:
        pool += retriever.keyword_hits(question, snap_idx, {h[0] for h in pool}, k=KEYWORD_EXTRA, qvec=qv)
    cands, positions, is_list, seen = [], [], [], set()   # (sentence, chunk rank, chunk index, chunk score)
    marks = []                                            # the footnote number a sentence starts with
    for rank, (ci, score, dense) in enumerate(pool):
        ch = retriever.cc[ci].chunk
        for s, pos, lst in _sentences(ch.text):
            mark = _FOOTNOTE.match(strip_links(s))
            plain = _FOOTNOTE.sub("", strip_links(s))     # "1Unlimited ATM access": a footnote mark
            if (len(plain) < 25 and not re.search(r"\d", plain)) or plain.lower() in seen or \
                    (skip is not None and skip.search(plain)):
                continue
            seen.add(plain.lower())
            cands.append((plain, rank, ci, score))
            positions.append(pos)
            is_list.append(lst)
            marks.append(mark.group(0) if mark else None)
    if not cands:
        return {"text": REFUSAL, "sources": [], "confident": False}

    if qv is not None:
        from embedder import DOC_PREFIX
        vecs = embedder.embed_docs_uncached([DOC_PREFIX.format(title=retriever.cc[ci].chunk.title or "none") + s
                                             for s, _, ci, _ in cands])
        # a two-part question's main sentence answers part 1; part 2 gets its own below
        sim = vecs @ (embedder.embed_query(part1) if part2 else qv)
    else:
        sim = np.zeros(len(cands))
    top_score = hits[0][1]
    today = retriever.dates[snap_idx]
    attr = _attr_phrase(question)
    business = bool(BUSINESS_Q.search(question)) and \
        any(_BUSINESS_URL.search(retriever.cc[ci].chunk.url) for _, _, ci, _ in cands)
    qyears = set(_YEAR.findall(question))
    if re.search(r"\bthis (calendar )?year\b", question, re.I):
        qyears.add(today[:4])              # "this year" is a hard filter; "last year" stays soft (older data)
    data_years = qyears | _relative_years(question, today)   # soft: the site may only have older data
    total_q = qtype is not None and (qtype[0] == "count" or
                                     (qtype[0] == "money" and (not TOTAL_WORDS_ONLY or TOTAL_Q.search(question))))
    recent_q = bool(RECENT_Q.search(question))      # "is it open yet?", "the newest branch"
    vals = []
    for n, (s, rank, ci, score) in enumerate(cands):
        overlap = len(qwords & _words(s)) / (len(qwords) or 1)
        val = float(sim[n]) + 0.5 * (score - top_score) + 0.15 * overlap
        if rank >= n_base:
            val -= KEYWORD_PENALTY         # keyword-only chunk: needs a clearly better sentence
        pub = _published(retriever, ci, snap_idx)
        if data_years and pub != "9999" and not _own_year(s, question) and \
                (pub[:4] < min(data_years) or (total_q and pub[:4] <= max(data_years))):
            # a 2024 release can't report "last year's" (2025) news, and a year's total is
            # only published after the year ends
            val -= OLD_ARTICLE_PENALTY
        elif YEAR_MISMATCH and data_years and _own_year(s, question) and not set(_YEAR.findall(s)) & data_years:
            val -= YEAR_MISMATCH           # "6,187 hours in 2024" doesn't answer "last year" (2025)
        elif total_q and data_years and PARTIAL_YEAR.search(s) and \
                (pub[:4] <= max(data_years) if pub != "9999" else today[:4] > max(data_years)):
            val -= OLD_ARTICLE_PENALTY     # "in 2025, Dupaco has already prevented ...": a mid-year count
        if qtype and qtype[1].search(s):
            val += 0.08
            if qtype[0] == "date" and FULL_DATE.search(s):
                val += 0.04                # "Oct. 1, 2024" answers "when" better than "in 2003"
            if qtype[0] == "date" and UPCOMING_Q.search(question):
                ds = _iso_dates(s)
                if ds and max(ds) < today:
                    val -= PAST_DEADLINE   # last cycle's deadline left on the page
        if qtype and qtype[0] == "where" and STREET.search(s):
            val += ADDRESS_BONUS
        if PDF_PENALTY and retriever.cc[ci].chunk.url.lower().endswith(".pdf"):
            val -= PDF_PENALTY
        if PLACEHOLDER.search(s):
            val -= 0.3
        if CALL_TO_ACTION.search(s) or (len(s) > 20 and s.upper() == s):
            val -= 0.05                    # "CLICK HERE FOR ...", "Learn more"
        if s[:1] in "“\"":            # quotations are opinions; facts sit in the narrative
            val -= 0.05
        if define and positions[n] <= 2 and _is_lead(retriever.cc[ci].chunk):
            val += 0.06                    # "what is X": the article's opening sentences define X
        if business and not _BUSINESS_URL.search(retriever.cc[ci].chunk.url):
            val -= SCOPE_PENALTY           # "the Business Rewards Visa late fee" is not the personal card's
        if attr is not None and attr.search(" ".join(_WORD.findall(s.lower()))):
            val += ATTR_BONUS              # names the asked fee/limit itself ("late payment fee")
        vals.append(val)
    if RERANK is not None:
        # dev pilot only (off when shipped): blend a reranker's score into the top candidates
        top_n = sorted(range(len(cands)), key=lambda n: -vals[n])[:RERANK_TOP]
        for n, r in zip(top_n, RERANK(part1 if part2 else question, [cands[n][0] for n in top_n])):
            vals[n] += RERANK_WEIGHT * r
    if LIST_Q.search(question) or (LIST_MORE and LIST_Q_MORE.search(question)):
        # "who is eligible / what are the benefits": a heading + bullet list in a top passage
        for n, (s, rank, ci, _) in enumerate(cands):
            if rank <= LIST_RANK and is_list[n]:
                vals[n] += 0.25
    order = sorted(range(len(cands)), key=lambda n: -vals[n])
    best_val = vals[order[0]]
    if DEBUG:
        for n in order[:8]:
            print(f"    {vals[n]:.3f} sim {float(sim[n]):.3f} r{cands[n][1]} {cands[n][0][:110]}")
    # near-ties go to the newest article: old releases restate facts that later changed
    near = [n for n in order if vals[n] >= best_val - (NEAR_TIE_TOTAL if (total_q and not _own_year(cands[n][0], question)) or recent_q
                                                      else NEAR_TIE)
            and (n == order[0] or cands[n][0].strip() != retriever.cc[cands[n][2]].chunk.title.strip())]
    best = max(near, key=lambda n: (_published(retriever, cands[n][2], snap_idx), vals[n]))
    if cands[best][0].rstrip().endswith("?") and best + 1 < len(cands) and cands[best + 1][2] == cands[best][2]:
        best += 1                          # FAQ: the matching question is followed by its answer
    need = [ok for rx, ok in NEEDS if rx.search(part1)
            and not (ok is RATE_NEED[1] and qtype and qtype[0] in ("phone", "person", "where", "date", "duration"))]

    fee_q = bool(FEE_Q.search(part1))
    q_fams = _families(part1) if FAMILY_GUARD else set()
    fee_mods = _fee_mods(part1)

    def fits(n: int) -> bool:
        s = cands[n][0]
        years = set(_YEAR.findall(s))
        ch = retriever.cc[cands[n][2]].chunk
        # "Is there a monthly fee?" is not answered by a rate-table row that only has a "$"
        # (a fee schedule's rows count: the page itself is about fees)
        if fee_q and not FEE_WORD.search(f"{s} {ch.title} {ch.section}"):
            return False
        # "What advisory fee ...?" is not answered by a sentence about some other (or no) fee
        if fee_q and fee_mods and not fee_mods & set(_WORD.findall(f"{s} {ch.title} {ch.section}".lower())):
            return False
        # "the minimum credit score for a car loan" is not answered by the home-loan minimum
        if q_fams:
            s_fams = _families(f"{s} {ch.title}")
            if s_fams and not s_fams & q_fams:
                return False
        # "How many used VTO in 2025?" is not answered by the 2024 count
        return all(ok.search(s) for ok in need) and not (qyears and years and not years & qyears)
    if (need or qyears or fee_q or q_fams) and not fits(best):
        alt = [n for n in order if vals[n] >= best_val - (0.4 if qtype and qtype[0] == "domain" else NEED_GAP) and fits(n)]
        if not alt:
            return {"text": REFUSAL, "sources": [], "confident": False}
        top = vals[alt[0]]
        best = max((n for n in alt if vals[n] >= top - NEAR_TIE),
                   key=lambda n: (_published(retriever, cands[n][2], snap_idx), vals[n]))
    s, rank, ci, _ = cands[best]
    recs = [cands[n][0] for n in range(len(cands)) if n != best and cands[n][2] == ci and not is_list[n]
            and "; " in cands[n][0] and not cands[n][0].startswith(("-", "*"))
            and s in cands[n][0] and len(cands[n][0]) < 300]
    if recs:
        s = min(recs, key=len)             # a card field answers with its card ("name; title; ext.")
    for o in _ORDINAL.finditer(question):
        # "the 78th annual meeting": an ordinal the passage never mentions is a different event
        if o.group(0).lower() not in retriever.cc[ci].chunk.text.lower():
            return {"text": REFUSAL, "sources": [], "confident": False}
    if hits[0][2] < UNSURE_DENSE:
        # weak retrieval: only answer if the sentence covers most of what was asked
        # ("lost debit card replacement" must not be answered with the debit-card rush fee)
        if _coverage(question, _words(s) | _words(retriever.cc[ci].chunk.title)) <= 0.5:
            return {"text": REFUSAL, "sources": [], "confident": False}
    text = s
    single_fact = qtype is not None and qtype[0] in ("count", "routing", "phone", "money", "date", "duration")
    if not single_fact and len(text) < 260:
        # open questions ("what is X", "who can join"): add the next-best sentence of the same
        # passage, or the sentence that follows the best one
        page = retriever.cc[ci].chunk.url
        follow = [n for n in order[:8] if n != best and retriever.cc[cands[n][2]].chunk.url == page
                  and vals[n] >= vals[best] - 2 * SECOND_GAP and cands[n][0] not in s and s not in cands[n][0]]
        nxt = best + 1 if best + 1 < len(cands) and cands[best + 1][2] == ci else None
        gap = 4 * SECOND_GAP if define else 2 * SECOND_GAP
        add = nxt if define and nxt is not None and vals[nxt] >= best_val - gap else \
            follow[0] if follow else nxt if nxt is not None and vals[nxt] >= best_val - gap else None
        if add is not None and cands[add][0] not in s and s not in cands[add][0]:
            pair = sorted((best, add))
            text = " ".join(cands[n][0] for n in pair)
    sib = [n for n in _siblings(question, best, cands, is_list, retriever)
           if (not fee_mods or fee_mods & set(_WORD.findall(cands[n][0].lower())))   # "foreign" fee rows only
           and " ".join(_WORD.findall(cands[n][0].lower())) not in " ".join(_WORD.findall(text.lower()))]
    if sib:
        text = "; ".join([text] + [cands[n][0] for n in sib])
    ref = _FOOTNOTE_REF.search(text) if FOOTNOTE_FOLLOW else None
    if ref:
        # "... as low as 6.25% APR will apply. 4": footnote 4 says who qualifies
        url = retriever.cc[ci].chunk.url
        note = next((n for n in range(len(cands)) if marks[n] == ref.group(1) and cands[n][0] not in text
                     and (retriever.cc[cands[n][2]].chunk.url == url or       # the same note on the rate sheet
                          retriever.on_page(cands[n][0][:40], url, snap_idx))), None)
        if note is not None:
            text = f"{(text[:ref.start(1)] + text[ref.end(1):]).rstrip()} {cands[note][0]}"
    extra_src = []
    if part2 and qv is not None:
        # "who won ..., and how much did they get?": the best sentence for the whole question
        # often answers only one half; add the sentence that best answers the other half
        p2v = embedder.embed_query(part2)
        sims2 = vecs @ p2v
        p2words = _qwords(part2)
        p2type = next(((name, ok) for name, rx, ok in QTYPES if rx.search(part2.lower())), None)
        have = [n for n in range(len(cands)) if cands[n][0] in text]
        cur = max((float(sims2[n]) for n in have), default=0.0)
        scored = []
        for n, (s2, _r2, _ci2, score2) in enumerate(cands):
            if cands[n][0] in text or (p2type and not p2type[1].search(s2)) or PLACEHOLDER.search(s2) or \
                    (skip is not None and skip.search(s2)):
                continue
            v = float(sims2[n]) + 0.5 * (score2 - top_score) + 0.15 * len(p2words & _words(s2)) / (len(p2words) or 1)
            scored.append((v, float(sims2[n]), n))
        if scored:
            v, sim2, n = max(scored)
            if cands[n][0].rstrip().endswith("?") and n + 1 < len(cands) and cands[n + 1][2] == cands[n][2]:
                n += 1                     # FAQ: the matching question is followed by its answer
            missing = p2type is not None and not p2type[1].search(text)   # e.g. no term length yet
            if DEBUG:
                print(f"    PART2 {part2!r} type {p2type and p2type[0]} cur {cur:.3f} cand {sim2:.3f} "
                      f"missing {missing}: {cands[n][0][:90]}")
            if sim2 >= cur + (-PART2_SLACK if missing else PART2_GAIN) and _words(cands[n][0]) - _words(text):
                text = f"{text} {cands[n][0]}"
                extra_src.append(retriever.cc[cands[n][2]].chunk.url)
    title = retriever.cc[ci].chunk.title
    if not retriever.cc[ci].published and title and not re.search(r"\d", title) and \
            not (_words(title) <= _words(text)) and "/rates" not in retriever.cc[ci].chunk.url:
        text = f"{title}: {text}"          # name the product page the answer comes from
    phone = retriever.main_phone(snap_idx) if HOWTO_Q.search(question) else None
    if phone and not re.search(r"\d{3}-\d{3}-\d{4}", text):
        text = f"{text} {CONTACT_LINE.format(phone=phone)}"
    src = [retriever.cc[ci].chunk.url] + extra_src + [retriever.cc[h[0]].chunk.url for h in hits[:3]
                                                      if retriever.cc[h[0]].chunk.url != retriever.cc[ci].chunk.url]
    return {"text": text, "sources": list(dict.fromkeys(src)), "confident": True}


PARTIAL_YEAR = re.compile(r"\b(already|so far|to date|year[- ]to[- ]date|through (the )?(first|second|third) quarter)\b", re.I)
_FOOTNOTE = re.compile(r"^\d(?=[A-Z][a-z])")
_FOOTNOTE_REF = re.compile(r"(?:[.!?]\s|[A-Za-z%*)])(\d)(?=\s|$|[;,.])")   # "apply. 4", "APY1 on"
_ROW = re.compile(r"^[^:;]{2,80}: ")
_LIST_SIB_Q = re.compile(r"%|\b(rates?|apy|apr|dividends?)\b", re.I)


def _siblings(question: str, best: int, cands, is_list, retriever, limit: int = 5) -> list[int]:
    """Other rows of the same table / cards of the same listing that match the question just as
    well as the chosen one ("wire transfers": incoming, outgoing and international rows; "lenders in
    Madison": every Madison card). A question that names one of them ("incoming wire") keeps one."""
    s, _r, ci, _sc = cands[best]
    text = retriever.cc[ci].chunk.text
    def kind(n):
        t = cands[n][0]
        if is_list[n] or cands[n][2] != ci:
            return None
        if "; " in t and not _ROW.match(t):
            return "record"
        return "row" if _ROW.match(t) and "|" in text else None
    k = kind(best)
    if k is None or _LIST_SIB_Q.search(question):
        return []
    matched = (_qwords(question) & _words(s)) - GENERIC - {"dupaco"}
    if not matched:
        return []
    need = len(matched) - (1 if len(matched) >= 2 else 0)     # "Domestic wire" row lacks "transfer"
    nums = set(re.findall(r"\d+", question)) & set(re.findall(r"\d+", s))   # "a 5 x 10 box" names its row
    named = (_qwords(question) - _words(s)) - GENERIC - {"dupaco"}       # "incoming, domestic and international"
    out, seen = [], {re.sub(r"\W+", " ", s.lower()).strip()}
    for n in range(len(cands)):
        t = cands[n][0]
        tw = _words(t)
        hit = len(matched & tw)
        if n == best or kind(n) != k or t in s or s in t or not nums <= set(re.findall(r"\d+", t)):
            continue
        if hit >= need or (hit and named & tw):
            norm = re.sub(r"\W+", " ", t.lower()).strip()
            if norm not in seen:               # the same row printed twice ("Incoming wire transfer $10.00")
                seen.add(norm)
                out.append(n)
    return out[:limit]


_PART2 = re.compile(r"(?:,\s*and|\band)\s+(?=(?:how|what|what's|who|where|when|which|is there|are there)\b)|"
                    r",\s*and\s+(?=(?:is|are|do|does|did|can|will)\b)|"
                    r"\?\s+(?:(?:and|also|plus)\s+)?(?=(?:how|what|what's|who|where|when|which|is|are|do|does|did|can|will)\b)|"
                    r"(?:[.!]\s+|\?\s+)also,?\s+(?=(?:how|what|what's|who|where|when|which|is|are|do|does|can)\b)", re.I)


def _second_part(question: str) -> str | None:
    """The second question in "Who won X, and how much did they get?" / "What can I do at
    a video teller? Do I need my card?"; None for a single question."""
    m = None
    for m in _PART2.finditer(question):
        pass
    if m is None:
        return None
    part = question[m.end():].strip()
    return part if len(part.split()) >= 3 else None


def _best_link(question: str, hits, retriever) -> str | None:
    """Best URL for 'what link / how do I ... online' questions: links in the top chunks
    (and the pages themselves) scored by word overlap with the question."""
    qw = _qwords(question) - {"link", "url", "website", "online", "page", "use", "see", "find"}
    best, best_score = None, 0.0
    for rank, (ci, *_rest) in enumerate(hits):
        ch = retriever.cc[ci].chunk
        options = [(ch.title, ch.url, True)] + [(m.group(1), m.group(2), False) for m in _LINK.finditer(ch.text)]
        for anchor, url, is_page in options:
            if "#" in url and url.split("#")[0].rstrip("/") == ch.url.rstrip("/"):
                continue
            if _DOC_EXT.search(url) and not re.search(r"\b(pdf|form|map|document)\b", question.lower()):
                continue
            words = _words(anchor) | _words(re.sub(r"[/._:-]", " ", url))
            # a matching link inside the page (e.g. the online application) beats the page itself
            score = len(qw & words) - 0.3 * rank + (0.5 if "dupaco.com" in url else 0) - (0.6 if is_page else 0)
            if score > best_score:
                best, best_score = url, score
    return best or (retriever.cc[hits[0][0]].chunk.url if hits else None)


# ---------------------------------------------------------------- list answers
LIST_ASK = re.compile(r"\b(what|which) (types|kinds|sorts) of\b|\b(types|kinds) of\b.*\b(offer|have|available)\b|"
                      r"\b(list|name) (all|the|every|each)\b|\bwho('s| is| are) on\b|"
                      r"\bwho (is|are) (dupaco'?s|the|on) (\w+ ){0,2}(team|leaders?|leadership|executives?|board)\b|"
                      r"\b(what|which) (\w+ ){0,3}(options|accounts|loans|services|cards|products|programs)\b.*"
                      r"\b(offer|have|available|choose|there)\b", re.I)
_OUTLINE_NOISE = re.compile(r"^(handpicked for you|frequently asked questions|faqs?|contact us|recent news|related|"
                            r"(\w+ ){0,3}rates|share this|resources|disclosures?|learn more|get started|apply|ready to|"
                            r"have questions|why you(['’]ll| will) love)\b|\?$", re.I)


_SMALL = {"a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "or", "at", "by", "vs.", "&"}


_EYEBROW_NOISE = re.compile(r"\b(TIP|TIPS|FEATURED|NEW|LEARN|FAQS?|SPOTLIGHT|WHY|HOW)\b")
_SHORT_WORDS = {"pay", "the", "and", "for", "of", "to", "you", "our", "new", "day", "fee", "car", "way", "at", "on", "in"}


def _headings(md: str) -> list[tuple[int, str]]:
    out = []
    gap = False                            # body text since the last heading
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line.strip())
        if m:
            text = strip_links(re.sub(r"[*_`]+", "", m.group(2))).strip()
            if text and not text.startswith("![") and re.search(r"[A-Za-z]{3}", text):
                level = len(m.group(1))
                if out and not gap and level <= 3 and out[-1][0] >= 5 and out[-1][1].upper() == out[-1][1] \
                        and len(out[-1][1].split()) <= 5 and not _EYEBROW_NOISE.search(out[-1][1]):
                    # "##### GAP PROTECTION" over "## Bridge the Gap": the eyebrow names the product
                    eyebrow = " ".join(w if len(w) <= 3 and w.lower() not in _SHORT_WORDS else w.capitalize()
                                       for w in out.pop()[1].split())
                    text = eyebrow
                out.append((level, text))
                gap = False
        elif line.strip():
            gap = True
    return out


def _parents(heads) -> list[int]:
    """Parent of each heading: the nearest earlier heading of a higher level (-1 = none)."""
    out = []
    for j, (level, _) in enumerate(heads):
        out.append(next((i for i in range(j - 1, -1, -1) if heads[i][0] < level), -1))
    return out


def _run(heads, kids: list[int]) -> list[int]:
    """The first run of 3+ consecutive same-level sub-headings: the product list of a hub
    page, not the marketing banners ("##### MEET SWEEPS", "## Helping Your Business...")."""
    kids = [j for j in kids if not _OUTLINE_NOISE.search(heads[j][1])
            and not (heads[j][1].upper() == heads[j][1] and len(heads[j][1]) > 3)
            # "Give yourself some credit": slogans run in lower case, product names are capitalised
            and len([w for w in heads[j][1].split() if w[:1].islower() and w not in _SMALL]) < 2]
    run: list[int] = []
    for j in kids:
        if run and heads[j][0] != heads[run[-1]][0]:
            if len(run) >= 3:
                return run
            run = []
        run.append(j)
    return run if len(run) >= 3 else []


def outline_answer(question: str, hits, retriever, page_text) -> dict | None:
    """'What types of loans do you offer?': the section headings of the matching page part.

    page_text(url) -> the page's cleaned markdown in the asked snapshot. Hub pages (short
    URLs: /loans, /personal/savings-account) are tried first. The anchor is the section of
    the hit, then its parent, then the page top; the answer is the anchor's sub-headings.
    A question that names one of those items ("what money market accounts...") is about
    that item, not the list, and is left to the sentence extractor."""
    if not LIST_ASK.search(question) or not hits or hits[0][2] < UNSURE_DENSE:
        return None
    qw = _qwords(question) - {"dupaco", "offer", "have", "type", "kind", "list", "option", "available", "what",
                              "which", "can", "choose", "from"}
    # the rarest asked-about word ("merchant", "ira") must appear in the page part listed
    toks = sorted({t for t in tokenize(question) if _stem(t) in qw and len(t) >= 3},
                  key=lambda t: -retriever.bm25.idf.get(t, 0.0))
    keys = toks[:3]
    pages, seen = [], set()
    for rank, (ci, _s, dense) in enumerate(hits[:6]):
        url = retriever.cc[ci].chunk.url
        if "/business" in url and not re.search(r"\b(business|llc|company|employer|nonprofit)", question, re.I):
            continue                      # "what types of loans" means personal loans
        if url not in seen and dense >= hits[0][2] - 0.08:
            seen.add(url)
            pages.append((url.rstrip("/").count("/"), rank, ci))
    for _depth, _rank, ci in sorted(pages)[:3]:
        ch = retriever.cc[ci].chunk
        md = page_text(ch.url)
        if not md:
            continue
        heads = _headings(md)
        if not heads:
            continue
        par = _parents(heads)
        anchor = (ch.section or "").split(" > ")[-1].strip()
        at = [i for i, (_, t) in enumerate(heads) if t == anchor]
        tries = (at[:1] + [par[at[0]]] if at and par[at[0]] >= 0 else at[:1]) + \
            [i for i, h in enumerate(heads) if h[0] == min(x[0] for x in heads)][:1]
        for i in dict.fromkeys(tries):
            if _OUTLINE_NOISE.search(heads[i][1]):
                continue                  # an FAQ's questions are not the product list
            items = _run(heads, [j for j in range(len(heads)) if par[j] == i])
            if not items or len(items) > 14:
                continue
            named = next((j for j in items if _words(heads[j][1]) <= qw), None)
            if named is not None:
                # the question names one item ("senior leadership", "money market accounts"):
                # list what is under it, or leave a leaf item to the sentence extractor
                items = _run(heads, [j for j in range(len(heads)) if par[j] == named])
                if not items:
                    return None
                i = named
            pool = _words(ch.title) | _words(heads[i][1])
            for j in items:
                pool |= _words(heads[j][1])
            if keys and not any(_stem(k) in pool or _stem(k).rstrip("s") in pool for k in keys):
                continue                  # the page is not about what was asked (merchant services -> join page)
            names = []
            for j in items:
                sub = [k for k in range(len(heads)) if par[k] == j]
                kids = [k for k in sub if not _OUTLINE_NOISE.search(heads[k][1])
                        and not re.match(r"(contact us|have questions|questions\b)", heads[k][1], re.I)]
                if len(kids) >= 3:
                    names += [heads[k][1] for k in kids]     # a marketing heading over the product list
                    continue
                # "Lisa Bowers (Chief People Officer)": a lone short sub-heading is a label
                label = heads[sub[0]][1] if len(sub) == 1 else ""
                names.append(f"{heads[j][1]} ({label})" if label and len(label) <= 60 and label.upper() != label
                             and not _OUTLINE_NOISE.search(label)
                             else heads[j][1])
            head = heads[i][1] if heads[i][0] > 1 else ch.title    # an H1 is a slogan; the title names the page
            return {"text": f"{head}: " + "; ".join(dict.fromkeys(names)) + ".", "sources": [ch.url], "confident": True}
    return None
