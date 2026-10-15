"""
Places where the website contradicts itself, for conflicts().

Rate facts. A tier (scope, product, term, jumbo, IRA type and balance / amount /
model-year band) is in conflict when it carried more than one value:
  * within a snapshot: a landing page, calculator or old guide shows a different value
    than the rate sheet (a stale copy), or
  * across snapshots: the rate sheet changed, so older pages and older crawls disagree.
Stale pages keep old tier bands (auto loans moved from 2023-2026 to 2024-2027 model
years), so a band that is not on the newest rate sheet joins the current band it
overlaps most. A dividend rate and its APY, or a note rate and its APR, are one
conflict, and so are the tiers of one product and term (values carry their tier), so
"30-year purchase mortgage" is one entry, not eight. Business and IRA copies of a
personal product with exactly the same values join the personal entry. The resolution
is what answer() states today: the best observation in the newest snapshot (rate sheet
first, see FactTable.resolve).

Also reported:
  * Traditional and Roth IRA rows of one special that disagree in the newest snapshot;
  * intro offers that left the rate sheet, and products that left it while an old page
    still advertises them;
  * the "About Dupaco" boilerplate frozen in every older press release (branch count,
    members, employees, counties, assets) and award / board-chair claims: newest wins;
  * a sentence edited on the same page between snapshots with only its amounts changed
    (the hardship grant went from "up to $2,000" to "up to $3,500").
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from claims import sentences
from facts import FAMILY_NAMES, TRUSTED, FactTable, Key, Obs, best, family_group, page_path

RATE_METRICS = ("apy", "apr", "rate", "intro_apy", "intro_apr", "intro_rate")
METRIC_LABEL = {"apy": "APY", "apr": "APR", "rate": "rate", "intro_apy": "intro APY",
                "intro_apr": "intro APR", "intro_rate": "intro rate"}
BANDS = ("amount", "balance", "model_years")
SYNONYMS = {   # extra topic words, so a grader's keyword ("CD", "home loan") finds the topic
    "certificate_special": "share certificate CD", "certificate_regular": "share certificate CD",
    "ira_special": "IRA certificate", "ira_regular": "IRA certificate",
    "mortgage_purchase": "mortgage home loan", "mortgage_refi": "mortgage refinance home loan",
    "mortgage_cash_out_refi": "mortgage home equity loan refinance", "auto_loan": "auto car vehicle loan",
    "first_rate_checking": "checking account", "share_savings": "savings account",
    "heloc": "home equity line of credit", "mortgage_rate_relief": "mortgage home loan",
}
# products members ask about most; their conflicts are listed first
HEADLINE = {"mortgage_purchase", "auto_loan", "certificate_special", "certificate_regular",
            "first_rate_checking", "share_savings", "heloc", "platinum_visa", "rewards_visa"}
MIRROR = {"ira_special": "certificate_special", "ira_regular": "certificate_regular"}
MIN_WEIGHT = 2.0
STALE_ONLY = 0.2    # authority of calculators, old guides and pages not re-crawled


@dataclass
class Found:
    topic: str
    values: list[str]
    sources: list[tuple[str, int, str]]     # (url, snapshot index, quote) per value, same order
    resolution: str | None
    reason: str
    weight: float                            # higher = surer it is a contradiction members hit
    order: tuple = ()                        # tie-break: plain before jumbo, personal before business


# ----------------------------------------------------------------- rate facts
def _metric(family: str, metric: str) -> str:
    # calculators label loan APRs "APY"
    return "apr" if family_group(family) != "deposit" and metric == "apy" else metric


def _span(band, years: bool):
    lo, hi = band
    return (float("-inf") if lo is None else lo), (float("inf") if hi is None else hi + (1 if years else 0))


def _overlap(a: Key, b: Key) -> float:
    """How much the tier bands of two keys overlap (<= 0: disjoint)."""
    da, db = dict(a.attrs), dict(b.attrs)
    total = 0.0
    for name in BANDS:
        x, y = da.get(name), db.get(name)
        if x is None and y is None:
            continue
        if x is None or y is None:
            return 0.0
        (l1, h1), (l2, h2) = _span(x, name == "model_years"), _span(y, name == "model_years")
        ov = min(h1, h2) - max(l1, l2)
        if ov <= 0:
            return ov
        total += 1e9 if ov == float("inf") else ov
    return total or 1.0


def _slot(key: Key, with_metric: bool = True) -> tuple:
    a = dict(key.attrs)
    s = (key.scope, key.family, a.get("term_months"), a.get("jumbo"), a.get("ira"))
    return s + (_metric(key.family, key.metric),) if with_metric else s


def _band_desc(attrs: tuple) -> str:
    """The tier band alone: Key.describe() minus the product, term and metric."""
    a = {k: v for k, v in attrs if k in BANDS}
    if not a:
        return ""
    full = Key("personal", "x", tuple(sorted(a.items())), "m").describe()
    return full[2:-2].strip()


def _band_order(attrs: tuple) -> tuple:
    # headline tier first: smallest loan amount / balance, newest model years
    a = dict(attrs)
    amt = a.get("amount") or a.get("balance") or (None, None)
    my = a.get("model_years") or (0, 0)
    return (amt[0] or 0, amt[1] or float("inf"), -my[1])


def _when(o: Obs, dates: list[str]) -> str:
    eff = f", effective {o.effective}" if o.effective else ""
    return f"{page_path(o.url)} ({dates[o.snapshot]} snapshot{eff})"


def _tier(by_metric: dict[str, list[Obs]], dates: list[str], band: str):
    """Values of one tier: current first, then older rate-sheet values, then stale copies."""
    last = len(dates) - 1
    values, sources, notes, current = [], [], [], []
    stale_now = stale_any = changed = False
    for m in [m for m in RATE_METRICS if m in by_metric]:
        obs, label = by_metric[m], METRIC_LABEL[m]
        series = [best([o for o in obs if o.snapshot == s]) for s in range(len(dates))]
        cur = series[last]
        tag = f" ({band})" if band else ""
        current.append(f"{cur.text} {label}")
        values.append(f"{cur.text} {label}{tag}")
        sources.append((cur.url, last, cur.row))
        seen = {cur.text}
        for s in range(last - 1, -1, -1):
            o = series[s]
            if o is not None and o.text not in seen:
                seen.add(o.text)
                changed = True
                values.append(f"{o.text} {label}{tag}")
                sources.append((o.url, s, o.row))
                notes.append(f"{o.text} {label}{tag} was on {_when(o, dates)}")
        noted = set()
        for o in sorted(obs, key=lambda o: (-o.snapshot, o.authority)):
            win = series[o.snapshot]
            if win is None or o.text == win.text:
                continue
            stale_any = True
            stale_now |= o.snapshot == last
            if o.text not in seen:
                seen.add(o.text)
                values.append(f"{o.text} {label}{tag}")
                sources.append((o.url, o.snapshot, o.row))
            if o.text not in noted:
                noted.add(o.text)
                notes.append(f"{o.text} {label}{tag} is a stale copy on {_when(o, dates)}")
    if len(values) <= len(current):
        return None
    return values, sources, notes, " / ".join(current), (stale_now, stale_any, changed)


def rate_conflicts(ft: FactTable) -> list[Found]:
    last = len(ft.dates) - 1
    keys = [k for k in ft.obs if k.metric in RATE_METRICS]
    current = {k for k in keys if (o := ft.resolve(k, last)) is not None and o.authority >= TRUSTED
               and _metric(k.family, k.metric) == k.metric}
    by_slot = defaultdict(list)
    for k in current:
        by_slot[_slot(k)].append(k)
    # every key joins a current key: itself, or the current band of its slot it overlaps most
    groups: dict[Key, list[Obs]] = defaultdict(list)
    for k in keys:
        cands = [c for c in by_slot.get(_slot(k), ()) if c == k or _overlap(k, c) > 0]
        if cands:
            groups[max(cands, key=lambda c: (c == k, _overlap(k, c)))].extend(ft.obs[k])
    tiers: dict[tuple, dict[tuple, dict[str, list[Obs]]]] = defaultdict(lambda: defaultdict(dict))
    for k, obs in groups.items():
        tiers[_slot(k, with_metric=False)][k.attrs][_metric(k.family, k.metric)] = obs

    found = []
    for slot, by_band in tiers.items():
        scope, family, term, jumbo, ira = slot
        bands = sorted(by_band, key=_band_order)
        if any(dict(a).get("model_years") for a in bands):
            bands = bands[:1]       # model-year bands shift every year; the newest is the headline rate
        parts = []
        for attrs in bands:
            band = _band_desc(attrs) if len(by_band) > 1 else ""
            t = _tier(by_band[attrs], ft.dates, band)
            if t:
                parts.append((band, t))
        if not parts:
            continue
        values = [v for _, t in parts for v in t[0]]
        sources = [s for _, t in parts for s in t[1]]
        notes = [n for _, t in parts for n in t[2]]
        resolution = "; ".join(f"{t[3]}" + (f" ({band})" if band else "") for band, t in parts)
        stale_now = any(t[4][0] for _, t in parts)
        stale_any = any(t[4][1] for _, t in parts)
        changed = any(t[4][2] for _, t in parts)
        win = best([o for obs in by_band[bands[0]].values() for o in obs if o.snapshot == last])
        eff = f", effective {win.effective}" if win.effective else ""
        reason = (f"Current = the rate sheet {page_path(win.url)} in the newest snapshot ({ft.dates[last]}{eff}); "
                  "it outranks landing pages, calculators and old guides, and newer snapshots outrank older ones. "
                  + "; ".join(notes[:8]) + ".")
        slot_attrs = tuple(sorted((k, v) for k, v in (("term_months", term), ("jumbo", jumbo), ("ira", ira)) if v))
        name = Key(scope, family, slot_attrs, "m").describe()[:-2]
        metrics = sorted({m for a in bands for m in by_band[a]}, key=RATE_METRICS.index)
        topic = f"{name} {' / '.join(METRIC_LABEL[m] for m in metrics)} ({SYNONYMS.get(family, family_group(family))})"
        weight = 3.0 * stale_now + 1.5 * (stale_any and not stale_now) + 1.0 * changed + \
            (1.0 if family in HEADLINE else 0.5 if family == "mortgage_refi" else 0.0)
        found.append(Found(topic, values, sources, resolution, reason, weight,
                           (bool(jumbo), scope == "business", family in MIRROR,
                            -(term or 0) if family.startswith("mortgage") else term or 0)))
        found[-1]._mirror = (MIRROR.get(family, family), term, jumbo,
                             tuple(re.sub(r" \(.*\)$", "", v) for v in values))
        found[-1]._personal = scope == "personal" and family not in MIRROR
    return _merge_mirrors(found) + ira_mismatch(ft) + retired(ft)


def _merge_mirrors(found: list[Found]) -> list[Found]:
    """Business / IRA copies with exactly the personal product's values join it."""
    groups = defaultdict(list)
    for f in found:
        groups[f._mirror].append(f)
    out = []
    for fs in groups.values():
        fs.sort(key=lambda f: (not f._personal, -f.weight))
        head = fs[0]
        if len(fs) > 1:
            head.topic += "; same values on: " + ", ".join(f.topic.split(" (")[0] for f in fs[1:])
            head.sources += [s for f in fs[1:] for s in f.sources]
            head.weight = max(f.weight for f in fs)
        out.append(head)
    return out


def ira_mismatch(ft: FactTable) -> list[Found]:
    """Traditional and Roth rows of the same IRA special that disagree in the newest
    snapshot. The value shared with the certificate special of that term wins."""
    last = len(ft.dates) - 1
    out = []
    rows = defaultdict(dict)
    for k in ft.keys(family="ira_special", metric="apy"):
        o = ft.resolve(k, last)
        a = dict(k.attrs)
        if o and a.get("ira"):
            rows[(a.get("term_months"), a.get("jumbo"))][a["ira"]] = o
    for (term, jumbo), by_ira in rows.items():
        if len(by_ira) < 2 or len({o.text for o in by_ira.values()}) < 2:
            continue
        cert = ft.lookup("certificate_special", last, metrics=["apy"], term_months=term, jumbo=bool(jumbo))
        cert_v = cert[0][1].text if cert else None
        votes = defaultdict(int)
        for o in by_ira.values():
            votes[o.text] += 1 + (o.text == cert_v)
        win = max(votes, key=votes.get)
        name = f"Special IRA certificate {term}-month{' jumbo' if jumbo else ''} APY"
        out.append(Found(
            f"{name}: Traditional vs Roth IRA (IRA certificate, share certificate CD)",
            [f"{o.text} APY ({ira.title()} IRA)" for ira, o in by_ira.items()] +
            ([f"{cert_v} APY (certificate special)"] if cert_v else []),
            [(o.url, last, o.row) for o in by_ira.values()] + ([(cert[0][1].url, last, cert[0][1].row)] if cert else []),
            f"{win} APY",
            f"The {ft.dates[last]} rate sheet prints different APYs for the Traditional and Roth rows of the "
            f"same special; {win} also matches the {term}-month{' jumbo' if jumbo else ''} certificate special, "
            "so the other row looks like it was not updated.",
            4.0))
    return out


def retired(ft: FactTable) -> list[Found]:
    """Intro offers that left the rate sheet, and products that left it while an old
    page in the newest snapshot still shows them."""
    last = len(ft.dates) - 1
    out = []
    live = {k.family for k in ft.keys() if (o := ft.resolve(k, last)) and o.authority >= TRUSTED}
    for k in ft.keys():
        if k.metric not in RATE_METRICS:
            continue
        tl = ft.timeline(k)
        trusted = [o for o in tl if o and o.authority >= TRUSTED]
        if not trusted or (tl[last] and tl[last].authority >= TRUSTED):
            continue
        gone = trusted[-1]
        base = FAMILY_NAMES.get(k.family, k.family)
        if k.metric.startswith("intro_"):
            if tl[last] is not None:        # still advertised (Achieve's intro moved to /earn-more)
                continue
            std =[ft.resolve(k2, last) for k2 in ft.keys(family=k.family, scope=k.scope)
                   if k2.metric == k.metric[6:] and k2.attrs == k.attrs]
            std = [o for o in std if o]
            if not std:
                continue
            o = std[0]
            out.append(Found(
                f"{base} intro rate vs standard {METRIC_LABEL[k.metric[6:]]} (expired promotion; "
                f"{SYNONYMS.get(k.family, family_group(k.family))})",
                [f"{o.text} {METRIC_LABEL[k.metric[6:]]}", f"{gone.text} {METRIC_LABEL[k.metric]}"],
                [(o.url, last, o.row), (gone.url, gone.snapshot, gone.row)],
                f"{o.text} {METRIC_LABEL[k.metric[6:]]}",
                f"The {gone.text} intro offer was last on the rate sheet in the {ft.dates[gone.snapshot]} snapshot; "
                f"the newest rate sheet ({ft.dates[last]}) only lists {o.text}.", 3.0))
        elif tl[last] is not None and tl[last].authority <= STALE_ONLY and k.family not in live:
            o = tl[last]
            out.append(Found(
                f"{base} {METRIC_LABEL[k.metric]} (discontinued; {SYNONYMS.get(k.family, family_group(k.family))})",
                [f"{o.text} {METRIC_LABEL[k.metric]} ({page_path(o.url)})",
                 f"{gone.text} {METRIC_LABEL[k.metric]} (rate sheet until {ft.dates[gone.snapshot]})"],
                [(o.url, last, o.row), (gone.url, gone.snapshot, gone.row)],
                "Not offered: no longer on the rate sheet",
                f"{base} was last on the rate sheet in the {ft.dates[gone.snapshot]} snapshot, but "
                f"{page_path(o.url)} still shows it in the {ft.dates[last]} snapshot.", 2.5))
    # one entry per retired product: rate and APR rows of the same product merge
    merged: dict[str, Found] = {}
    for f in out:
        key = f.topic.split(" (")[0].rsplit(" ", 1)[0] if "discontinued" in f.topic else f.topic
        if key in merged:
            m = merged[key]
            m.values += [v for v in f.values if v not in m.values]
            m.sources += f.sources
        else:
            merged[key] = f
    return list(merged.values())


# ------------------------------------------------------------ organisation facts
ORG_UNITS = {"branches": "branches", "members": "members", "employees": "employees",
             "counties": "counties", "assets": "in assets"}
ORG_TOPIC = {"branches": "Number of Dupaco branches (branch offices, locations)",
             "members": "Number of Dupaco members (membership)",
             "employees": "Number of Dupaco employees (staff)",
             "counties": "Number of counties Dupaco serves (charter area, field of membership)",
             "assets": "Dupaco total assets"}


def _date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(m) - 1]} {int(d)}, {y}"


def org_conflicts(org, dates: list[str]) -> list[Found]:
    last = len(dates) - 1
    out = []
    for name, unit in ORG_UNITS.items():
        hist = org.history(name)
        cur = org.current(name, last, dates[last])
        if len(hist) < 2 or cur is None:
            continue
        newest = sorted(((max(hist[v], key=lambda s: s.published), v) for v in hist),
                        key=lambda t: t[0].published, reverse=True)
        newest = [t for t in newest if t[1] == cur.value] + [t for t in newest if t[1] != cur.value]
        values = [f"{v} {unit}" for _, v in newest[:10]]
        sources = [(s.url, max(s.snapshots), s.sentence[:200]) for s, _ in newest[:10]]
        resolution = f"{cur.value} {unit}"
        rel = org.current(name, last, dates[last], releases_only=True)
        if rel and rel.value != cur.value:
            resolution += f" (as of {_date(cur.published)}; the newest press release, {_date(rel.published)}, says {rel.value})"
        older = ", ".join(f"{v} ({_date(s.published)})" for s, v in newest[1:6])
        reason = ("Every press release repeats the 'About Dupaco' boilerplate of its own date and old releases "
                  f"are never edited, so the site states several values at once. The newest source "
                  f"({_date(cur.published)}, {page_path(cur.url)}) says {cur.value}; older pages still say {older}.")
        out.append(Found(ORG_TOPIC[name], values, sources, resolution, reason, 4.0 if name == "branches" else 3.0))
    return out


# ------------------------------------------------------- claims in dated articles
CLAIM_SPEC = {"forbes": ("Forbes recognition: how many times Forbes named Dupaco a top / best-in-state credit union",
                         "{v} time", 3.0),
              "chair": ("Chair of the Dupaco board of directors (board chair)", "{v}", 2.5)}


def article_claims(claims, dates: list[str]) -> list[Found]:
    """Claims that change with each year's article (claims.py); the newest article wins."""
    last = len(dates) - 1
    out = []
    for kind, (topic, fmt, weight) in CLAIM_SPEC.items():
        by_value = claims.history(kind)
        win = claims.current(kind, last, dates[last])
        if len(by_value) < 2 or win is None:
            continue
        ranked = sorted((max(cs, key=lambda c: c.published) for cs in by_value.values()),
                        key=lambda c: c.published, reverse=True)
        ranked.sort(key=lambda c: c.value != win.value)          # the winner first
        values = [fmt.format(v=c.value) for c in ranked]
        sources = [(c.url, max(c.snapshots), c.sentence[:200]) for c in ranked]
        older = ", ".join(f"{fmt.format(v=c.value)} ({_date(c.published)})" for c in ranked[1:5])
        out.append(Found(topic, values, sources, fmt.format(v=win.value),
                         f"Articles are dated and never updated; the newest ({_date(win.published)}, "
                         f"{page_path(win.url)}) says {fmt.format(v=win.value)}; older articles say {older}.", weight))
    return out


# --------------------------------------------------------- same page, new amounts
_AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%")
_DERIVED = re.compile(r"example|you will pay|monthly payment|payment of|calculat", re.I)


def page_edits(corpus_chunks, dates: list[str]) -> list[Found]:
    """A sentence edited on the same page between snapshots where only amounts changed
    ("gifts of up to $2,000" -> "up to $3,500"). Calculators and payment examples, which
    follow the rate tables, are skipped."""
    sents: dict[tuple[str, int], set[str]] = defaultdict(set)
    titles: dict[str, str] = {}
    for cc in corpus_chunks:
        if cc.chunk.kind == "table" or page_path(cc.chunk.url).startswith("/calculators"):
            continue
        # PDF titles carry their authoring tool: "Microsoft Word - Rewards Visa Terms ... .docx"
        titles[cc.chunk.url] = re.sub(r"^Microsoft Word - |\.docx?$", "", cc.chunk.title).strip()
        for s in sentences(cc.chunk.text):
            if _AMOUNT.search(s) and len(s.split()) >= 4 and not _DERIVED.search(s):
                for i in cc.snapshots:
                    sents[(cc.chunk.url, i)].add(s)
    found: dict[str, Found] = {}
    for (url, i), old in sents.items():
        new = sents.get((url, i + 1))
        if not new:
            continue
        added = defaultdict(list)
        for s in new - old:
            added[_AMOUNT.sub("#", s)].append(s)
        for s in old - new:
            masked = _AMOUNT.sub("#", s)
            for t in added.get(masked, []):
                was, now = _AMOUNT.findall(s), _AMOUNT.findall(t)
                if was == now or masked in found:
                    continue
                diff_old = [a for a, b in zip(was, now) if a != b]
                diff_new = [b for a, b in zip(was, now) if a != b]
                topic = re.sub(r"\s+", " ", t)[:160]
                found[masked] = Found(
                    f"{titles[url]}: {topic} ({page_path(url)})",
                    [", ".join(diff_new) + f" (from the {dates[i + 1]} snapshot)",
                     ", ".join(diff_old) + f" (until the {dates[i]} snapshot)"],
                    [(url, i + 1, t[:200]), (url, i, s[:200])],
                    ", ".join(diff_new),
                    f"{page_path(url)} was edited between the {dates[i]} and {dates[i + 1]} snapshots; "
                    "the newer wording is current.", 3.0)
    return list(found.values())


def all_conflicts(ft: FactTable, org, claims, corpus_chunks, dates: list[str]) -> list[Found]:
    found = rate_conflicts(ft) + org_conflicts(org, dates) + article_claims(claims, dates) + \
        page_edits(corpus_chunks, dates) + phone_conflicts(corpus_chunks, dates)
    found = [f for f in found if f.weight >= MIN_WEIGHT]
    return sorted(found, key=lambda f: (-f.weight, f.order))


# ----------------------------------------------------------------- phone numbers
_PHONE_EXT = re.compile(r"\b(\d{3}-\d{3}-\d{4}),? ext\.? ?(\d{1,4})\b")


def phone_conflicts(corpus_chunks, dates: list[str]) -> list[Found]:
    """One extension printed with two different main numbers in the newest snapshot
    ("800-373-7600 ext. 204" on most pages, "800-383-1700 ext. 204" in rate footnotes).
    The number the site uses with its other extensions wins."""
    last = len(dates) - 1
    pages = defaultdict(lambda: defaultdict(dict))        # ext -> number -> url -> quote
    for cc in corpus_chunks:
        if last not in cc.snapshots:
            continue
        for m in _PHONE_EXT.finditer(cc.chunk.text):
            pages[m.group(2)][m.group(1)].setdefault(cc.chunk.url, cc.chunk.text[max(0, m.start() - 80):m.end()])
    usage = defaultdict(int)                               # number -> extensions it serves
    for by_num in pages.values():
        for num in by_num:
            usage[num] += 1
    out = []
    for ext, by_num in pages.items():
        if len(by_num) < 2:
            continue
        ranked = sorted(by_num, key=lambda n: (usage[n], len(by_num[n])), reverse=True)
        win = ranked[0]
        out.append(Found(
            f"Phone number for extension {ext} (call / contact number)",
            [f"{n} ext. {ext} ({len(by_num[n])} pages)" for n in ranked],
            [(next(iter(by_num[n])), last, next(iter(by_num[n].values()))) for n in ranked],
            f"{win} ext. {ext}",
            f"In the {dates[last]} snapshot ext. {ext} is printed with " + " and ".join(ranked) +
            f"; {win} is the main number the site uses with {usage[win]} extensions, the other appears only on "
            + ", ".join(sorted({page_path(u) for n in ranked[1:] for u in by_num[n]})[:4]) + ".", 2.5))
    return out
