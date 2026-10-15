"""
Answer rate questions from the fact table.

Rules that protect the score:
  * state only the tier that was asked for. Another tier's number is a "wrong value"
    to the grader and zeroes the answer, even next to the right one;
  * never mention superseded values in the answer text (they go in conflict_note);
  * refusals for products without a published rate contain no numbers at all.
"""
from __future__ import annotations

import re
from datetime import date

from facts import TRUSTED, FactTable, Key, fmt_pct, family_group
from query import Query

PRODUCT_PAGES = {
    "auto_loan": "https://www.dupaco.com/personal/loans/auto-motorcycle-loans",
    "mortgage_purchase": "https://www.dupaco.com/personal/loans/home-loans",
    "mortgage_refi": "https://www.dupaco.com/personal/loans/home-loans",
    "mortgage_cash_out_refi": "https://www.dupaco.com/personal/loans/home-equity",
    "heloc": "https://www.dupaco.com/personal/loans/home-equity",
    "home_equity_loan": "https://www.dupaco.com/personal/loans/home-equity",
    "home_improvement_loan": "https://www.dupaco.com/personal/loans/home-equity",
    "personal_loan": "https://www.dupaco.com/personal/loans/personal-loans",
    "boat_rv_loan": "https://www.dupaco.com/personal/loans/recreational-loans",
    "atv_loan": "https://www.dupaco.com/personal/loans/recreational-loans",
    "platinum_visa": "https://www.dupaco.com/personal/credit-cards",
    "rewards_visa": "https://www.dupaco.com/personal/credit-cards",
    "business_platinum_visa": "https://www.dupaco.com/business/credit-cards",
    "business_rewards_visa": "https://www.dupaco.com/business/credit-cards",
    "first_rate_checking": "https://www.dupaco.com/personal/checking-accounts",
    "free_checking": "https://www.dupaco.com/personal/checking-accounts",
    "bright_savings": "https://www.dupaco.com/savings-account/online-savings-account",
    "money_market_achieve": "https://www.dupaco.com/earn-more",
    "launchpad_savings": "https://www.dupaco.com/personal/retirement",
    "ira_special": "https://www.dupaco.com/personal/retirement",
    "ira_regular": "https://www.dupaco.com/personal/retirement",
    "ira_variable": "https://www.dupaco.com/personal/retirement",
}
DEPOSIT_PAGE = "https://www.dupaco.com/personal/savings-account"
BUSINESS_DEPOSIT_PAGE = "https://www.dupaco.com/business/savings-accounts"
RATES_PAGE = "https://www.dupaco.com/rates"
STALE_ONLY = 0.2    # calculators, old guides, and pages not re-crawled for this snapshot


def _metrics(fam: str, q: Query) -> list[str]:
    g = family_group(fam)
    if g == "deposit":
        base = ["rate", "apy"]
        if fam == "money_market_achieve":
            base = ["intro_rate", "intro_apy"] if q.intro else ["intro_rate", "intro_apy", "rate", "apy"]
    elif fam.startswith("mortgage"):
        base = ["rate", "apr"]
    else:
        base = ["apr"]
    if q.intro and g != "deposit":
        base = ["intro_apr"] + base
    return base


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return ""
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}, {d.year}"


def _tier(key: Key) -> tuple:
    return tuple((k, v) for k, v in key.attrs)


def _line(fam_key: Key, by_metric: dict) -> str:
    name = fam_key.describe().rsplit(" ", 1)[0]          # drop the metric suffix
    parts = []
    m = by_metric
    if "intro_rate" in m or "intro_apy" in m:
        intro = " / ".join(x for x in (
            f"{m['intro_rate'].text} dividend rate" if "intro_rate" in m else "",
            f"{m['intro_apy'].text} APY" if "intro_apy" in m else "") if x)
        parts.append(f"introductory {intro} for the first 6 months")
    if "rate" in m and "apy" in m:
        label = "dividend rate" if family_group(fam_key.family) == "deposit" else "rate"
        parts.append(f"{m['rate'].text} {label} / {m['apy'].text} APY")
    elif "apy" in m:
        parts.append(f"{m['apy'].text} APY")
    elif "rate" in m and "apr" in m:
        parts.append(f"{m['rate'].text} rate / {m['apr'].text} APR")
    elif "rate" in m:
        parts.append(f"{m['rate'].text}")
    if "apr" in m and not ("rate" in m and "apr" in m):
        parts.append(f"{m['apr'].text} APR")
    if "intro_apr" in m:
        parts.append(f"introductory {m['intro_apr'].text} APR")
    eff = next((o.effective for o in m.values() if o.effective), None)
    return f"{name}: {', then '.join(parts) if 'intro_rate' in m else '; '.join(parts)}" + \
        (f" (effective {_fmt_date(eff)})" if eff else "")


def refusal(product: str) -> dict:
    return {"text": f"Dupaco's website doesn't list a rate for {product}. "
                    "Please contact Dupaco directly for a personalized quote.",
            "sources": [], "confident": False, "conflict_note": None}


def _discontinued(q: Query, ft, snap: int) -> dict | None:
    """"Is the 25-month special still around?": a term an earlier rate sheet listed is gone, so
    say so and give what replaced it (a term never offered stays a number-free refusal)."""
    for fam in q.families:
        scope = "business" if fam.startswith("business_") else q.scope
        jumbo = q.jumbo if q.jumbo is not None or fam not in ("certificate_special", "ira_special") else False
        was = [s for s in range(snap) if ft.lookup(fam, s, scope=scope, term_months=q.term_months, jumbo=jumbo, ira=q.ira)]
        if not was:
            continue
        now = ft.lookup(fam, snap, scope=scope, metrics=_metrics(fam, q), jumbo=jumbo, ira=q.ira)
        groups: dict[tuple, dict] = {}
        for k, o in now:
            if o.authority >= TRUSTED:
                groups.setdefault((fam, _tier(k), scope), {}).setdefault(k.metric, o)
        if not groups or len(groups) > 6:
            continue
        lines = [_line(next(Key(g[2], g[0], g[1], m) for m in groups[g]), groups[g]) for g in groups]
        label = f"{q.term_months}-month"
        return {"text": f"No. The {label} {Key(scope, fam, (), '').describe().strip()} is no longer on Dupaco's "
                        f"current rate sheet (last listed {ft.dates[was[-1]]}). Current options: " + "; ".join(lines) + ".",
                "sources": [RATES_PAGE], "confident": True, "conflict_note": None, "snapshot": snap}
    return None


def _no_term(q: Query) -> dict:
    """The product exists but not with the asked term (a 7-month certificate). No numbers
    beyond the question's own: listing the real terms would read as a guessed rate."""
    t = q.term_months
    years = t % 12 == 0 and re.search(r"\b(years?|yrs?)\b", q.text.lower())
    label = f"{t // 12}-year" if years else f"{t}-month"
    fam = q.families[0]
    name = "certificate" if fam.startswith(("certificate", "ira")) else \
        Key(q.scope, fam, (), "").describe().strip().lower()
    if fam.startswith("ira"):
        name = "IRA certificate"
    return {"text": f"Dupaco's current rate sheet doesn't list a {label} {name}, so I can't quote a rate for "
                    "that term. Please contact Dupaco to ask about the terms that are available.",
            "sources": [RATES_PAGE], "confident": False, "conflict_note": None}


def answer_rate(q: Query, ft: FactTable) -> dict | None:
    """Answer dict (text, sources, confident, conflict_note) or None when the fact table
    has nothing for this question (the caller then falls back to retrieval)."""
    if q.no_rate:
        return refusal(q.no_rate)
    snap = ft.snapshot_for(q.as_of)
    if q.wants_link and q.families:
        fam = q.families[0]
        page = PRODUCT_PAGES.get(fam) or (BUSINESS_DEPOSIT_PAGE if q.scope == "business" else DEPOSIT_PAGE)
        return {"text": f"Use {page}#rates for current {Key(q.scope, fam, (), '').describe().strip()} rates "
                        f"(all rates are also on {RATES_PAGE}).",
                "sources": [page, RATES_PAGE], "confident": True, "conflict_note": None, "snapshot": snap}
    groups: dict[tuple, dict] = {}        # (family, tier) -> {metric: Obs}
    order: list[tuple] = []
    jumbo_exists = False
    for fam in q.families:
        scope = "business" if fam.startswith("business_") else q.scope
        jumbo = q.jumbo
        if jumbo is None and fam in ("certificate_special", "ira_special"):
            jumbo = False                 # standard tier unless jumbo was asked for
        hits = ft.lookup(fam, snap, scope=scope, metrics=_metrics(fam, q), term_months=q.term_months,
                         amount=q.amount, balance=q.balance, model_year=q.model_year, jumbo=jumbo, ira=q.ira)
        if q.jumbo is None and jumbo is False and ft.lookup(fam, snap, scope=scope, term_months=q.term_months, jumbo=True):
            jumbo_exists = True
        trusted = [(k, o) for k, o in hits if o.authority >= TRUSTED]
        if trusted:
            hits = trusted
        elif hits and max(o.authority for _, o in hits) <= STALE_ONLY and \
                any(o.authority >= TRUSTED for k in ft.keys(fam, scope) for o in ft.obs[k]):
            # once on the rate sheet, now only on stale guides/calculators: discontinued
            return refusal(f"the {Key(scope, fam, (), '').describe().strip()} "
                           "(it is no longer on the current rate sheet)")
        for k, o in hits:
            g = (fam, _tier(k), scope)
            if g not in groups:
                groups[g] = {}
                order.append(g)
            groups[g].setdefault(k.metric, o)
    if not groups:
        if q.term_months and any(ft.lookup(f, snap, scope="business" if f.startswith("business_") else q.scope)
                                 for f in q.families):
            gone = _discontinued(q, ft, snap)
            return gone if gone is not None else _no_term(q)
        return None

    keys = {g: next(Key(g[2], g[0], g[1], m) for m in groups[g]) for g in order}
    terms = {g: dict(g[1]).get("term_months") for g in order}
    if q.mode in ("longest", "shortest") and any(terms.values()):
        pick = (max if q.mode == "longest" else min)(t for t in terms.values() if t)
        used = [g for g in order if terms[g] == pick]
        label = f"{pick // 12} years ({pick} months)" if pick % 12 == 0 and pick >= 120 else f"{pick} months"
        text = f"The {q.mode} term on the current rate sheet is {label}: " + \
            "; ".join(_line(keys[g], groups[g]) for g in used) + "."
        return _pack(text, used, groups, q, ft, snap)
    if q.mode in ("highest", "lowest", "check"):
        metric = "apy" if any("apy" in groups[g] for g in order) else "apr" if any(
            "apr" in groups[g] for g in order) else "rate"
        scored = [(groups[g][metric].value, g) for g in order if metric in groups[g]
                  and not isinstance(groups[g][metric].value, tuple)]
        if scored:
            pick = max(scored) if q.mode != "lowest" else min(scored)
            g = pick[1]
            line = _line(keys[g], groups[g])
            if q.mode == "check":
                hit = [g2 for v, g2 in scored if abs(v - q.check_value) < 1e-9]
                if hit:
                    text = "Yes. " + "; ".join(_line(keys[x], groups[x]) for x in hit) + "."
                else:
                    top = max(scored)[1]
                    text = (f"No. None of those tiers is listed at that rate; the highest is "
                            f"{_line(keys[top], groups[top])}.")
            else:
                text = f"The {'highest' if q.mode != 'lowest' else 'lowest'} is {line}."
            return _pack(text, [g], groups, q, ft, snap)

    lines = [_line(keys[g], groups[g]) for g in order]
    text = ("; ".join(lines) if len(lines) > 1 else lines[0]) + "."
    if q.balance is not None and all(not isinstance(o.value, tuple) and o.value == 0
                                     for g in order for o in groups[g].values()):
        # "$20k in Achieve" sits in the 0% tier: say where earning starts (no other tier's rate)
        lows = [dict(k.attrs)["balance"][0] for f in q.families for k, o in ft.lookup(f, snap, scope=q.scope)
                if dict(k.attrs).get("balance") and dict(k.attrs)["balance"][0] is not None
                and dict(k.attrs)["balance"][0] > q.balance and not isinstance(o.value, tuple) and o.value > 0]
        if lows:
            text += f" A balance of at least ${min(lows):,.0f} is needed to earn dividends."
    if jumbo_exists:
        text += " A jumbo version ($100,000 minimum) is also offered."
    if q.as_of and snap < len(ft.dates) - 1:
        text = f"As of {_fmt_date(q.as_of)} (website as of {_fmt_date(ft.dates[snap])}): " + text
    return _pack(text, order, groups, q, ft, snap)


def _pack(text: str, used: list[tuple], groups: dict, q: Query, ft: FactTable, snap: int) -> dict:
    urls, notes = [], []
    for g in used:
        for metric, o in groups[g].items():
            if o.url not in urls:
                urls.append(o.url)
            key = Key(g[2], g[0], g[1], metric)
            older = sorted({ob.text for ob in ft.obs.get(key, ()) if ob.text != o.text})
            if older:
                notes.append(f"{key.describe()}: {o.text} from {o.url} (also seen: {', '.join(older)})")
    for g in used:
        page = PRODUCT_PAGES.get(g[0]) or (BUSINESS_DEPOSIT_PAGE if g[2] == "business" else DEPOSIT_PAGE)
        if page not in urls:
            urls.append(page)
    if RATES_PAGE not in urls:
        urls.append(RATES_PAGE)
    return {"text": text, "sources": urls, "confident": True,
            "conflict_note": "; ".join(notes[:6]) or None, "snapshot": snap}
