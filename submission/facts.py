"""
Fact table: every rate-table row on every page, in every snapshot, as typed facts.

A fact key is (scope, family, attrs, metric):
  scope   "personal" | "business"
  family  canonical product, e.g. "certificate_special", "auto", "mortgage_purchase"
  attrs   tier qualifiers: term_months, jumbo, ira (roth/traditional), balance range,
          loan-amount range, model-year band, intro
  metric  "rate" (dividend / note rate), "apy", "apr", "intro_apr", "points", "payment",
          "min_open", "min_balance", ...

The same key can be observed on several pages and in several snapshots. Resolving a
key for a snapshot picks the observation from the most authoritative page (the official
rate sheet beats product pages, which beat landing pages, which beat calculators and
blog posts), so stale copies never win; they become conflicts instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

# ------------------------------------------------------------------ page authority
_CANONICAL = {"/rates", "/business-rates"}
_PRODUCT = {
    "/personal/savings-account", "/personal/retirement", "/personal/checking-accounts",
    "/personal/credit-cards", "/personal/loans/home-loans", "/personal/loans/home-equity",
    "/personal/loans/auto-motorcycle-loans", "/personal/loans/personal-loans",
    "/personal/loans/recreational-loans", "/savings-account/online-savings-account",
    "/business/savings-accounts", "/business/checking-accounts", "/business/credit-cards",
}
_LOW = {"/homebuying-guidance", "/mortgage-refi", "/switch-and-earn"}
STALE_FETCH_DAYS = 14
TRUSTED = 0.6       # authority at or above this: rate sheet, product pages, aliases


def page_path(url: str) -> str:
    return urlparse(url).path.rstrip("/").lower() or "/"


def family_group(family: str) -> str:
    if family.endswith("_visa"):
        return "card"
    if family.endswith(("_loan", "_refi")) or family.startswith(("mortgage", "heloc")):
        return "loan"
    return "deposit"


# product pages are only trusted for their own topic: /personal/checking-accounts also
# carried (stale) loan tables in snapshots 01-04
_TOPIC_BY_PATH = [("/personal/loans", "loan"), ("/personal-loans", "loan"), ("/loans", "loan"),
                  ("/learn/master-your-money/homeownership", "loan"), ("/personal/credit-cards", "card"),
                  ("/business/credit-cards", "card"), ("/personal", "deposit"), ("/savings", "deposit"),
                  ("/checking", "deposit"), ("/business", "deposit")]


def authority(url: str, family: str | None = None) -> float:
    p = page_path(url)
    if p in _CANONICAL:
        return 1.0
    if family and not p.endswith("-lp"):
        topic = next((t for pre, t in _TOPIC_BY_PATH if p.startswith(pre)), None)
        if topic and topic != family_group(family):
            return 0.5
    if p in _PRODUCT:
        return 0.8
    if p.startswith("/calculators") or p in _LOW:
        return 0.2
    if p.startswith(("/learn", "/newsroom", "/business/resources", "/resources", "/20", "/dupacodaily")):
        return 0.1
    if p.startswith("/hubfs"):
        return 0.3
    if p.endswith("-lp") or p in {"/earn-more", "/makethemost-lp"}:
        return 0.4
    return 0.6          # aliases / duplicates of product pages (/checking, /loans, .../auto-loans/ ...)


# ------------------------------------------------------------------- cell parsing
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_RANGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%?\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*%")
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?")
_BARE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")
_EFFECTIVE = re.compile(r"effective (?:as of )?(\d{1,2})/(\d{1,2})/(\d{2,4})", re.I)


def fmt_pct(x: float) -> str:
    """4.1 -> '4.10%', 6.125 -> '6.125%' (two decimals minimum, three when present)."""
    s = f"{x:.3f}"
    return (s[:-1] if s.endswith("0") else s) + "%"


def parse_pct(cell: str, bare_ok: bool = False):
    """'4.10%2' -> 4.10 ; '12.50% – 19.50%2' -> (12.5, 19.5) ; '7.29' -> 7.29 (if bare_ok)."""
    c = cell.replace("\\", "").strip()
    m = _RANGE.match(c)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = _PCT.search(c)
    if m and c.find(m.group(0)) < 8:         # leading percentage; ignore later mentions
        return float(m.group(1))
    m = _BARE.match(c)
    if m and bare_ok:
        return float(m.group(1))
    return None


def money_values(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "") + (f".{m.group(2)}" if m.group(2) else ""))
            for m in _MONEY.finditer(text)]


def effective_date(text: str) -> str | None:
    m = _EFFECTIVE.search(text)
    if not m:
        return None
    mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    y = y + 2000 if y < 100 else y
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def column_role(h: str) -> str:
    h = h.lower().replace("\\", "")
    if "payment" in h:
        return "payment"
    if "point" in h:
        return "points"
    if "min" in h and "open" in h:
        return "min_open"
    if "min" in h and ("earn" in h or "balance" in h):
        return "min_balance"
    if "max" in h and "balance" in h:
        return "max_balance"
    yld = "yield" in h or "apy" in h
    if "after promo" in h:
        return "apy" if yld else "rate"
    if re.search(r"\b(1st|first|intro)", h) and "month" in h:
        return "intro_apy" if yld else "intro_rate"
    if yld:
        return "apy"
    if "apr" in h or "annual percentage rate" in h:
        return "apr"
    if "dividend" in h or h.strip() in ("rate", "interest rate"):
        return "rate"
    return "label"


# --------------------------------------------------------------- classification
FAMILY_RULES: list[tuple[str, str]] = [   # (regex on label text, family) first match wins
    (r"achieve", "money_market_achieve"),
    (r"investor", "money_market_investors_choice"),
    (r"high[- ]yield", "high_yield_savings"),
    (r"you[- ]name[- ]it", "you_name_it_savings"),
    (r"holiday club", "holiday_club_savings"),
    (r"bright savings", "bright_savings"),
    (r"launchpad", "launchpad_savings"),
    (r"health savings|\bhsa\b|saving account rates", "hsa"),
    (r"share savings", "share_savings"),
    (r"1st rate|first rate|whatever your checking|day-to-day money management", "first_rate_checking"),
    (r"free checking", "free_checking"),
    (r"mortgage rate relief", "mortgage_rate_relief"),
    (r"purchase:", "mortgage_purchase"),
    (r"(cash[- ]out|home equity).*refinance:|refinance:.*(cash[- ]out|home equity)", "mortgage_cash_out_refi"),
    (r"refinance:", "mortgage_refi"),
    (r"heloc|home equity line", "heloc"),
    (r"home equity loan|loan – standard|loan - standard", "home_equity_loan"),
    (r"home improvement", "home_improvement_loan"),
    (r"atv|snowmobile|jet ski", "atv_loan"),
    (r"boat|\brv\b", "boat_rv_loan"),
    (r"model year|auto|motorcycle", "auto_loan"),
    (r"personal loan", "personal_loan"),
    (r"business platinum", "business_platinum_visa"),
    (r"business rewards", "business_rewards_visa"),
    (r"platinum", "platinum_visa"),
    (r"rewards", "rewards_visa"),
    # business 1st Rate Checking tiers sit under a bare "Regular" tab on /business-rates
    (r"\bregular\b.*\bbalance\b.*\$[\d,]", "first_rate_checking"),
    (r"(roth|traditional).*special|special iras?", "ira_special"),
    (r"(roth|traditional).*\$|variable iras?", "ira_variable"),
    (r"roth|traditional|regular iras?", "ira_regular"),
    (r"special", "certificate_special"),
    (r"\d+\s*month|regular", "certificate_regular"),
]

FAMILY_NAMES = {
    "money_market_achieve": "Achieve Money Market", "money_market_investors_choice": "Investor's Choice Money Market",
    "high_yield_savings": "High-Yield Savings", "you_name_it_savings": "You Name It Savings",
    "holiday_club_savings": "Holiday Club Savings", "bright_savings": "Bright Savings",
    "launchpad_savings": "Launchpad Savings", "hsa": "Health Savings Account", "share_savings": "Share Savings",
    "first_rate_checking": "1st Rate Checking", "free_checking": "Free Checking",
    "mortgage_rate_relief": "Mortgage Rate Relief Program", "mortgage_purchase": "Fixed-rate mortgage (purchase)",
    "mortgage_cash_out_refi": "Cash-out / home equity refinance", "mortgage_refi": "Fixed-rate mortgage (refinance)",
    "heloc": "Home Equity Line of Credit (HELOC)", "home_equity_loan": "Home Equity Loan (fixed)",
    "home_improvement_loan": "Home Improvement Loan", "atv_loan": "ATV, Snowmobile & Jet Ski Loan",
    "boat_rv_loan": "Boat & RV Loan", "auto_loan": "Auto & Motorcycle Loan", "personal_loan": "Personal Loan",
    "business_platinum_visa": "Business Platinum Visa", "business_rewards_visa": "Business Rewards Visa",
    "platinum_visa": "Platinum Visa", "rewards_visa": "Rewards Visa", "ira_special": "Special IRA certificate",
    "ira_variable": "Variable IRA", "ira_regular": "Regular IRA certificate",
    "certificate_special": "Certificate special", "certificate_regular": "Regular certificate",
}


# one balance tier only; some pages print it ('$5 – $1,000'), others don't
SINGLE_TIER = {"bright_savings", "launchpad_savings", "holiday_club_savings", "free_checking"}


def classify(label: str) -> str | None:
    low = label.lower()
    for rx, fam in FAMILY_RULES:
        if re.search(rx, low):
            return fam
    return None


def _range(text: str):
    """Money range in a tier label: '$10,000 – $24,999.99' / '$25,000+' / '$100,000 or less'."""
    t = text.replace("\\", "")
    m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:-|–|—|to)\s*\$?\s?([\d,]+(?:\.\d+)?)", t)
    if m:
        return (float(m.group(1).replace(",", "")), float(m.group(2).replace(",", "")))
    m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:\+|and (?:up|above|over))", t)
    if m:
        return (float(m.group(1).replace(",", "")), None)
    m = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*or less", t)
    if m:
        return (None, float(m.group(1).replace(",", "")))
    return None


def attributes(family: str, label: str) -> dict:
    a: dict = {}
    low = label.lower().replace("–", "-").replace("—", "-")
    if family.startswith("mortgage"):
        m = re.search(r"(\d+)\s*-?\s*year", low)
        if m:
            a["term_months"] = int(m.group(1)) * 12
    else:
        m = re.search(r"(\d+)\s*-?\s*(months?|mos?\b)", low) or re.search(r"(\d+)\s*-?\s*(years?)\b", low)
        if m:
            a["term_months"] = int(m.group(1)) * (12 if m.group(2).startswith("year") else 1)
    m = re.search(r"(20\d\d)\s*-\s*(\d\d)\b", low)
    if m and ("model year" in low or family in ("auto_loan", "atv_loan")):
        y1, y2 = int(m.group(1)), 2000 + int(m.group(2))
        a["model_years"] = (min(y1, y2), max(y1, y2))
    elif re.search(r"(20\d\d) model year \(and older\)", low):
        a["model_years"] = (0, int(re.search(r"(20\d\d)", low).group(1)))
    r = None if family in SINGLE_TIER else _range(label)
    if r:   # whole-dollar lower bound, so '$0.00 - $99.99' and '$0.01 - $99.99' are one tier
        lo, hi = r
        r = (None if lo is None else float(int(lo)), None if hi is None else round(hi, 2))
        a["amount" if "loan" in family or family.startswith("mortgage") else "balance"] = r
    if "jumbo" in low:
        a["jumbo"] = True
    for ira in ("roth", "traditional"):
        if ira in low:
            a["ira"] = ira
    if "intro" in low:
        a["intro"] = True
    return a


# ------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Key:
    scope: str
    family: str
    attrs: tuple
    metric: str

    def describe(self) -> str:
        a = dict(self.attrs)
        name = FAMILY_NAMES.get(self.family, self.family)
        parts = [("Business " if self.scope == "business" and not name.startswith("Business") else "") + name]
        if a.get("ira"):
            parts.append(a["ira"].title())
        if "term_months" in a:
            t = a["term_months"]
            parts.append(f"{t // 12}-year" if self.family.startswith("mortgage") else f"{t}-month")
        if a.get("jumbo"):
            parts.append("jumbo")
        if "model_years" in a:
            lo, hi = a["model_years"]
            parts.append(f"{hi} model year and older" if lo == 0 else f"{lo}-{hi} model years")
        for k in ("balance", "amount"):
            if k in a:
                lo, hi = a[k]
                money = (lambda x: f"${x:,.0f}" if x == int(x) else f"${x:,.2f}")
                parts.append(f"{money(lo)} and up" if hi is None else f"{money(hi)} or less" if lo is None
                             else f"{money(lo)}-{money(hi)}")
        if a.get("intro"):
            parts.append("intro")
        return " ".join(parts) + f" {self.metric.upper() if self.metric in ('apy', 'apr') else self.metric}"


@dataclass
class Obs:
    value: float | tuple
    snapshot: int
    url: str
    authority: float
    effective: str | None
    fetched_at: str
    row: str

    @property
    def text(self) -> str:
        v = self.value
        return f"{fmt_pct(v[0])}–{fmt_pct(v[1])}" if isinstance(v, tuple) else fmt_pct(v)


@dataclass
class FactTable:
    obs: dict[Key, list[Obs]] = field(default_factory=dict)
    dates: list[str] = field(default_factory=list)

    def add(self, key: Key, ob: Obs):
        self.obs.setdefault(key, []).append(ob)

    def resolve(self, key: Key, snapshot: int) -> Obs | None:
        """Best observation of `key` in `snapshot`.

        Trusted pages (rate sheet, product pages and their aliases) outrank landing pages,
        calculators and posts. Among trusted pages the newest "effective as of" date wins:
        in snapshot 04 the official /business-rates sheet still showed June certificate
        specials while /business/savings-accounts already had the July ones."""
        return best([o for o in self.obs.get(key, ()) if o.snapshot == snapshot])

    def keys(self, family: str | None = None, scope: str | None = None, metric: str | None = None):
        return [k for k in self.obs if (family is None or k.family == family)
                and (scope is None or k.scope == scope) and (metric is None or k.metric == metric)]

    def timeline(self, key: Key) -> list[Obs | None]:
        return [self.resolve(key, s) for s in range(len(self.dates))]

    def snapshot_for(self, as_of: str | None) -> int:
        """Latest snapshot taken on or before `as_of` (the newest one when None)."""
        if not as_of:
            return len(self.dates) - 1
        ok = [i for i, d in enumerate(self.dates) if d <= as_of[:10]]
        return ok[-1] if ok else 0

    def lookup(self, family: str, snapshot: int, scope: str = "personal", metrics=None,
               term_months=None, amount=None, balance=None, model_year=None, jumbo=None, ira=None):
        """Resolved facts of `family` matching the given tier constraints, best first.

        A constraint left as None matches any tier. Tier bands can differ between pages
        (stale pages keep old model-year bands), so overlapping keys compete on authority
        and only the strongest observation per (metric, term, jumbo, ira) survives."""
        hits = []
        for key in self.obs:
            if key.family != family or key.scope != scope or (metrics and key.metric not in metrics):
                continue
            a = dict(key.attrs)
            if term_months is not None and a.get("term_months") not in (None, term_months):
                continue
            if jumbo is not None and bool(a.get("jumbo")) != jumbo:
                continue
            if ira is not None and a.get("ira") not in (None, ira):
                continue
            if not _inside(model_year, a.get("model_years")) or not _inside(amount, a.get("amount")) \
                    or not _inside(balance, a.get("balance")):
                continue
            ob = self.resolve(key, snapshot)
            if ob is not None:
                hits.append((key, ob))
        best: dict[tuple, tuple] = {}
        for key, ob in hits:
            a = dict(key.attrs)
            slot = (key.metric, a.get("term_months"), a.get("jumbo"), a.get("ira"),
                    None if model_year is None else "my", None if amount is None else "amt",
                    None if balance is None else "bal")
            if model_year is None:
                slot += (a.get("model_years"),)
            if amount is None:
                slot += (a.get("amount"),)
            if balance is None:
                slot += (a.get("balance"),)
            # most authoritative, then the tightest band: $90k fits every "$X or less" tier
            rank = (ob.authority, -_width(a, amount, balance, model_year))
            if slot not in best or rank > best[slot][2]:
                best[slot] = (key, ob, rank)
        return [(k, o) for k, o, _ in sorted(best.values(), key=lambda v: (-v[2][0], v[0].metric, str(v[0].attrs)))]

    def variants(self, key: Key) -> dict[str, list[Obs]]:
        """Every distinct value ever observed for `key`, with its observations."""
        out: dict[str, list[Obs]] = {}
        for o in self.obs.get(key, ()):
            out.setdefault(o.text, []).append(o)
        return out


def best(cands: list[Obs]) -> Obs | None:
    """The observation that wins among competing ones (see FactTable.resolve)."""
    def rank(o):
        trusted = o.authority >= TRUSTED
        return (trusted, (o.effective or "") if trusted else "", o.authority, o.fetched_at, o.effective or "")
    return max(cands, key=rank) if cands else None


def _width(attrs: dict, amount, balance, model_year) -> float:
    w = 0.0
    for name, given in (("amount", amount), ("balance", balance), ("model_years", model_year)):
        band = attrs.get(name)
        if given is None or band is None:
            continue
        lo, hi = band
        w += float("inf") if hi is None else hi - (lo or 0)
    return w


def _inside(x, band) -> bool:
    if x is None or band is None:
        return True
    lo, hi = band
    return (lo is None or x >= lo) and (hi is None or x <= hi)


# ------------------------------------------------------------------- extraction
def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def parse_table(text: str):
    """(header cells, body rows as cell lists, effective date) from a table chunk."""
    table, _, tail = text.partition("\n\n")
    rows = [r for r in table.splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return None
    header = _cells(rows[0])
    body = [_cells(r) for r in rows[1:] if not re.match(r"^\|\s*:?-", r.strip())]
    return header, body, effective_date(tail or text)


def _fetch_penalty(fetched_at: str, taken_on: str) -> float:
    try:
        lag = (date.fromisoformat(taken_on) - date.fromisoformat(fetched_at[:10])).days
    except ValueError:
        return 1.0
    return 0.5 if lag > STALE_FETCH_DAYS else 1.0


def extract(chunk, url: str) -> list[tuple[Key, float | tuple, str, str | None]]:
    """(key, value, raw row, effective date) for every value cell of a table chunk."""
    parsed = parse_table(chunk.text)
    if not parsed:
        return []
    header, body, eff = parsed
    roles = [column_role(h) for h in header]
    if not any(r in ("rate", "apy", "apr", "intro_rate", "intro_apy") for r in roles):
        return []
    sec = chunk.section
    out = []
    for cells in body:
        if len(cells) != len(header):
            continue
        label_cells = [c for c, r in zip(cells, roles) if r == "label"]
        row_label = " ".join(label_cells)
        # section + header + row: the section carries the product when rows only name a
        # tier ('$25,000+'), and a header can name it too ('| Platinum Credit Card | APR |')
        fam = classify(" ".join([sec, *[h for h, r in zip(header, roles) if r == "label"], row_label]))
        if fam is None:
            continue
        scope = "business" if (page_path(url).startswith("/business") or fam.startswith("business_")
                               or "business" in f"{sec} {row_label}".lower()) else "personal"
        attrs = attributes(fam, f"{row_label} {sec}" if fam.startswith(("ira", "certificate")) else row_label)
        is_intro = bool(re.search(r"intro", row_label, re.I))
        for c, r in zip(cells, roles):
            if r == "label":
                continue
            if r in ("min_open", "min_balance", "max_balance", "payment"):
                vals = money_values(c)
                if not vals:
                    continue
                v = vals[0]
            elif r == "points":
                v = parse_pct(c, bare_ok=True)
            else:
                v = parse_pct(c, bare_ok=r in ("apr", "rate"))
            if v is None:
                continue
            metric = "intro_apr" if (is_intro and r in ("apr", "rate")) else r
            key = Key(scope, fam, tuple(sorted((k, v2) for k, v2 in attrs.items() if k != "intro")), metric)
            out.append((key, v, "| " + " | ".join(cells) + " |", eff))
    return out


def build_fact_table(corpus_chunks, dates: list[str]) -> FactTable:
    ft = FactTable(dates=list(dates))
    for cc in corpus_chunks:
        ch = cc.chunk
        if ch.kind != "table":
            continue
        rows = extract(ch, ch.url)
        if not rows:
            continue
        for s in sorted(cc.snapshots):
            fetched = cc.fetched_at.get(s, dates[s])
            penalty = _fetch_penalty(fetched, dates[s])
            for key, v, row, eff in rows:
                ft.add(key, Obs(v, s, ch.url, authority(ch.url, key.family) * penalty, eff, fetched, row))
    return ft
