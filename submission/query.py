"""
Rule-based question parser: member question -> structured rate lookup.

No LLM: synonyms come from the site's own product names plus common member wording
("CD", "car loan", "five years", "drive-thru"...). Unknown questions parse to
intent None and are left to retrieval.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ number words
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD = re.compile(r"\b(" + "|".join(_TENS) + r")(?:[\s-](" + "|".join(list(_UNITS)[1:10]) + r"))?\b|\b("
                      + "|".join(_UNITS) + r")\b")


def words_to_digits(text: str) -> str:
    def sub(m):
        if m.group(1):
            return str(_TENS[m.group(1)] + (_UNITS[m.group(2)] if m.group(2) else 0))
        return str(_UNITS[m.group(3)])
    t = re.sub(r"\bhalf[- ]a[- ]year\b", "6 months", text)
    t = re.sub(r"\ba (year|month)\b", r"1 \1", t)
    return _NUMWORD.sub(sub, t)


# ------------------------------------------------------------------- dates
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS["sept"] = 9
_MON = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DATE_FULL = re.compile(_MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)", re.I)
_DATE_MONTH = re.compile(_MON + r",?\s+(20\d\d)", re.I)
_DATE_ISO = re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})\b")


def date_in_text(q: str) -> tuple[str | None, str]:
    """('2026-05-31', question-without-the-date) for 'as of May 2026' style mentions."""
    for rx, conv in ((_DATE_ISO, lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
                     (_DATE_FULL, lambda m: f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"),
                     (_DATE_MONTH, lambda m: (lambda y, mo: f"{y}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}")(
                         int(m.group(2)), _MONTHS[m.group(1).lower()]))):
        m = rx.search(q)
        if m:
            return conv(m), q[:m.start()] + " " + q[m.end():]
    return None, q


# ------------------------------------------------------------------ product lexicon
# members name the car, not the loan ("a 2023 Honda for 72 months")
CAR_BRANDS = ("honda", "toyota", "ford", "chevy", "chevrolet", "gmc", "dodge", "ram", "jeep", "nissan", "subaru",
              "hyundai", "kia", "mazda", "volkswagen", "vw", "bmw", "audi", "mercedes", "lexus", "tesla", "buick",
              "cadillac", "chrysler", "acura", "infiniti", "volvo", "mitsubishi", "lincoln", "harley")
# (regex over the normalised question, families). Order matters: specific before generic.
PRODUCTS: list[tuple[str, list[str]]] = [
    (r"\b(adjustable|arm|[0-9]+/1)\b.*\b(mortgage|rate|loan|arm)\b|\b(5|7|10)/1\b|\barms?\b", ["NO_RATE:adjustable-rate mortgage"]),
    (r"\bconstruction\b", ["NO_RATE:construction loan"]),
    (r"\b(lot|land)\s+loan", ["NO_RATE:lot or land loan"]),
    (r"\bbridge loan", ["NO_RATE:bridge loan"]),
    (r"\bstudent\b.*\b(rate|apr|interest)\b|\b(rate|apr|interest)\b.*\bstudent\b", ["NO_RATE:student loan"]),
    (r"\bbusiness\b.*\b(loan|line of credit|lending)\b|\b(equipment|commercial|sba)\b.*\bloan", ["NO_RATE:business loan"]),
    (r"\bshare[- ]secured|\bcertificate[- ]secured", ["NO_RATE:share-secured loan"]),
    (r"\bjumbo (mortgage|home loan|loan)s?\b|\b(mortgage|home loan)s?\b.*\bjumbo\b", ["NO_RATE:jumbo mortgage"]),
    (r"\bmortgage rate relief|\brate relief\b", ["mortgage_rate_relief"]),
    (r"\banalysis checking\b", ["NO_RATE:Business Analysis Checking"]),
    (r"\bachieve\b", ["money_market_achieve"]),
    (r"\binvestor'?s?\s*choice\b", ["money_market_investors_choice"]),
    (r"\bmoney market\b", ["money_market_investors_choice", "money_market_achieve", "high_yield_savings"]),
    (r"\bhigh[- ]?yield\b", ["high_yield_savings"]),
    (r"\byou[- ]?name[- ]?it\b", ["you_name_it_savings"]),
    (r"\b(holiday|christmas) club\b", ["holiday_club_savings"]),
    (r"\bbright savings\b|\bonline[- ](only )?savings\b|\bbright\b", ["bright_savings"]),
    (r"\blaunch ?pad\b", ["launchpad_savings"]),
    (r"\bhsa\b|\bhealth savings\b", ["hsa"]),
    (r"\b(1st|first) rate\b", ["first_rate_checking"]),
    (r"\bfree checking\b", ["free_checking"]),
    (r"\bchecking\b", ["first_rate_checking", "free_checking"]),
    (r"\bheloc\b|\bhome equity line\b|\bequity line\b", ["heloc"]),
    (r"\bcash[- ]out\b(?!.*\b(points?|rewards?|visa|card)\b)", ["mortgage_cash_out_refi"]),
    (r"\bhome equity\b", ["home_equity_loan"]),
    (r"\bhome improvement\b|\bremodel", ["home_improvement_loan"]),
    (r"\b(atv|snowmobile|jet ?ski|utv|side[- ]by[- ]side)s?\b", ["atv_loan"]),
    (r"\b(boat|rv|camper|motorhome|recreational)s?\b", ["boat_rv_loan"]),
    (r"\b(auto|car|vehicle|truck|suv|motorcycle|van|pickup)s?\b|" + r"\b(" + "|".join(CAR_BRANDS) + r")\b", ["auto_loan"]),
    (r"\brefinanc\w*\b.*\b(mortgage|home|house)\b|\b(mortgage|home|house)\b.*\brefinanc", ["mortgage_refi"]),
    (r"\bmortgages?\b|\bhome loans?\b|\bhouse\b|\bhome purchase\b|\b(15|20|30)[- ]?y(ea)?r\b.*\bfixed\b",
     ["mortgage_purchase"]),
    (r"\bpersonal loans?\b|\bsignature loan", ["personal_loan"]),
    (r"\bbusiness platinum\b", ["business_platinum_visa"]),
    (r"\bbusiness rewards\b", ["business_rewards_visa"]),
    (r"\bbusiness\b.*\b(credit )?cards?\b", ["business_platinum_visa", "business_rewards_visa"]),
    (r"\bplatinum\b", ["platinum_visa"]),
    (r"\brewards?\b.*\b(card|visa)\b|\b(visa|card)\b.*\brewards?\b", ["rewards_visa"]),
    (r"\bbalance transfer\b", ["platinum_visa"]),
    (r"\bcredit cards?\b|\bvisas?\b", ["platinum_visa", "rewards_visa", "business_platinum_visa", "business_rewards_visa"]),
    (r"\b(roth|traditional)?\s*iras?\b", ["IRA"]),
    (r"\b(certificates?|cds?|share certificates?|term shares?)\b|\b\d+[- ]?month (fixed )?(jumbo )?specials?\b",
     ["CERTIFICATE"]),
    (r"\bshare savings\b|\bsavings accounts?\b|\bregular savings\b|\bbasic savings\b|\bsavings\b", ["share_savings"]),
]

# "the most the hardship grant pays" is no rate lookup: a highest/lowest question must name an account or loan
PRODUCT_WORDS = re.compile(r"\b(certificates?|cds?|savings|checking|money market|mortgages?|loans?|heloc|home equity|iras?|visa|credit cards?|cards?|refinanc\w*|accounts?)\b")
RATE_WORDS = re.compile(r"\b(rates?|apys?|aprs?|yields?|interest|dividends?|pay(s|ing)?|earn(s|ing)?|percent|charges?|specials?|deals?|promotions?)\b|%")
LINK_WORDS = re.compile(r"\b(link|url|web ?page|website|web site)\b|\bwhere (can|do|should) i (see|find|look)\b")
LIST_WORDS = re.compile(r"\b(all|list|every|each|options|terms|tiers|offer|what are|rates)\b")
HIGHEST = re.compile(r"\b(highest|best|top|max(imum)?|most)\b")
LONGEST = re.compile(r"\blongest\b|\bmax(imum)? (loan )?(term|length)\b|"
                     r"\bhow long (can|could) (i|you|a|the|my)\b.*\b(term|loan|finance|borrow|go)\b|\bhow many (months|years) can i\b")
SHORTEST = re.compile(r"\bshortest\b|\bmin(imum)? (term|length)\b")
LOWEST = re.compile(r"\b(lowest|cheapest|minimum|least)\b")
# questions that mention a product and "rate" but ask something else (who to call, how to
# qualify, what credit score) go to retrieval, not to a rate lookup
NOT_A_LOOKUP = re.compile(r"\bcredit scores?\b|\bqualify\b|\brequirements?\b|\bdocuments?\b|\bwho (do|should|can) i (call|contact|talk)\b|"
                          r"\bphone\b|\bcontact\b|\bhow (do|can) i (apply|lock)\b|\block (in|my)\b|\bclosing costs?\b|"
                          r"\bcash[- ]?back\b|\bredemptions?\b|\bredeem\b|\brewards? points\b|\bpoints\b|\bminimum loan\b|"
                          r"\bloan size\b|\bget (anything|something|money) back\b|\b(annual|late|foreign transaction) fees?\b|"
                          r"\b(monthly|service|maintenance|overdraft|nsf|atm|wire) (service )?(fees?|charges?)\b|"
                          r"\bcredit limit\b|\bmini[- ]?max\b|\b(most|max(imum)?|how much) (i|you|we|my \w+) (can|could) borrow\b|"
                          r"\b(grants?|scholarships?|hardship|foundation|donations?|awards?)\b|\bborrow up to\b|\bloan (amounts?|limits?|maximums?)\b|\bterms (and|&) conditions\b|\bterms of (use|service)\b|"
                          r"\bbonus\b|\bpromo(tion(al)?)? code\b|\bget the \$|\$\d+ (bonus|cash|offer|reward)|\bpayout\b")
SPECIAL_TERMS = {10, 17, 25}
JUMBO_LIMIT = 832_750     # conforming limit on the rate sheet; larger mortgages have no published rate


@dataclass
class Query:
    text: str
    families: list[str] = field(default_factory=list)
    no_rate: str | None = None                # product described on the site but without a rate
    scope: str = "personal"
    metrics: list[str] = field(default_factory=list)
    term_months: int | None = None
    amount: float | None = None
    balance: float | None = None
    model_year: int | None = None
    jumbo: bool | None = None
    ira: str | None = None
    intro: bool = False
    special: bool | None = None
    mode: str = "single"                      # single | list | highest | lowest | check
    check_value: float | None = None          # "does any tier pay 0.35%?"
    as_of: str | None = None
    is_rate_question: bool = False
    wants_link: bool = False
    product_list: bool = False


def _amounts(t: str) -> list[float]:
    out = []
    for m in re.finditer(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(k|thousand|million|m)?\b"
                         r"|\b(\d{1,3}(?:,\d{3})+|\d+)\s*(k|thousand|million)\b(?!\s*miles)", t):
        num = (m.group(1) or m.group(4)).replace(",", "")
        val = float(num + (f".{m.group(2)}" if m.group(2) else ""))
        mult = (m.group(3) or m.group(5) or "").lower()
        val *= 1000 if mult in ("k", "thousand") else 1e6 if mult in ("m", "million") else 1
        out.append(val)
    return out


def parse(question: str, as_of: str | None = None) -> Query:
    q = Query(text=question)
    text_date, rest = date_in_text(question)
    q.as_of = as_of or text_date
    t = " " + words_to_digits(rest.lower().replace("–", "-").replace("—", "-").replace("’", "'")) + " "
    t = re.sub(r"\b(apple|google|samsung) pay\b|\bpay (off|bills?|down)\b|\bbill pay\b", " ", t)

    # products: the first (most specific) matching rule wins
    fams: list[str] = []
    for rx, fs in PRODUCTS:
        if re.search(rx, t):
            if fs[0].startswith("NO_RATE:"):
                q.no_rate = fs[0][8:]
            else:
                fams = list(fs)
            break
    if not fams and re.search(r"\b(19[89]\d|20[0-3]\d)\b", t) and re.search(r"\bfinanc(e|ed|es|ing)\b|\bapr\b|\bloan\b", t) \
            and not re.search(r"\b(mortgage|home|house|heloc|certificates?|savings|ira|business|student)\b", t):
        fams = ["auto_loan"]               # "financing a 2025 Equinox": a model year means a vehicle
    q.scope = "business" if re.search(r"\bbusiness\b", t) else "personal"

    # tiers
    m = re.search(r"\b(\d{1,3})\s*-?\s*(months?|mos?|mo\.)(?![a-z])", t)
    if m:
        q.term_months = int(m.group(1))
    else:
        m = re.search(r"\b(\d{1,2})\s*-?\s*(years?|yrs?|yr\.)(?![a-z])(?!\s*old)", t)
        if m and not re.search(r"\bmodel[- ]year", t[max(0, m.start() - 12):m.end() + 12]):
            q.term_months = int(m.group(1)) * 12
    m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", t)
    if m and (fams == ["auto_loan"] or re.search(r"model|vehicle|car|truck|suv|motorcycle|atv|snowmobile|boat|jet ?ski|van|"
                        + "|".join(CAR_BRANDS), t)):
        q.model_year = int(m.group(1))
    elif re.search(r"\bnew (car|vehicle|truck|suv|motorcycle)\b", t):
        q.model_year = int((q.as_of or "2026")[:4])
    amts = _amounts(t)
    if amts:
        loanish = any(f.endswith(("_loan", "_refi")) or f.startswith(("mortgage", "heloc")) for f in fams)
        if loanish:
            q.amount = max(amts)
            if any(f.startswith("mortgage") for f in fams) and q.amount > JUMBO_LIMIT:
                q.no_rate, fams = "a jumbo mortgage (above the conforming loan limit)", []
        else:
            q.balance = amts[0]
    if re.search(r"\b(not|no|non)[- ](the |a )?jumbo\b|\bnon-?jumbo\b|\b(not|no|n.t)\b[^.?]{0,30}\bjumbo\b", t):
        q.jumbo = False                   # "not the jumbo one, the regular one"
    else:
        q.jumbo = True if re.search(r"\bjumbo\b", t) else None
    m = re.search(r"\b(roth|traditional)\b", t)
    q.ira = m.group(1) if m else None
    if re.search(r"\broth\b", t) and re.search(r"\btraditional\b", t):
        q.ira = None                       # "the same for Roth and Traditional?": both
    q.intro = bool(re.search(r"\b(intro(ductory)?|promo(tional)?|teaser|balance transfer)\b", t))
    q.special = True if re.search(r"\bspecials?\b", t) else None

    # metrics
    if re.search(r"\bapy\b|\byield\b", t):
        q.metrics = ["apy"]
    elif re.search(r"\bdividend rate\b", t):
        q.metrics = ["rate"]
    elif re.search(r"\bapr\b|\bannual percentage rate\b", t):
        q.metrics = ["apr"]

    # mode
    pm = re.search(r"(\d+(?:\.\d+)?)\s*%", rest)
    plain = re.sub(r"\bmini[- ]?max\b", " ", t)
    plain = re.sub(r"\bmost (recent|recently|popular|common|likely)\b|\bat most\b", " ", plain)
    if pm and re.search(r"\b(any|does|do|is there|are there)\b", t):
        q.mode, q.check_value = "check", float(pm.group(1))
    elif LONGEST.search(plain) or SHORTEST.search(plain):
        # a term in the question is a reference point ("a blog said 18 months"), not the tier
        q.mode, q.term_months = ("longest" if LONGEST.search(plain) else "shortest"), None
    elif HIGHEST.search(plain):
        q.mode = "highest"
    elif LOWEST.search(plain) and not re.search(r"\bminimum (deposit|balance|to open)", t):
        q.mode = "lowest"

    # resolve generic certificate / IRA into concrete families
    concrete = []
    for f in fams:
        if f == "CERTIFICATE":
            if q.special or q.jumbo or (q.term_months in SPECIAL_TERMS):
                concrete.append("certificate_special")
            elif q.term_months:
                concrete.append("certificate_regular")
            else:
                concrete += ["certificate_special", "certificate_regular"]
        elif f == "IRA":
            if re.search(r"\bvariable\b", t):
                concrete.append("ira_variable")
            elif q.special or q.jumbo or (q.term_months in SPECIAL_TERMS):
                concrete.append("ira_special")
            elif q.term_months:
                concrete.append("ira_regular")
            else:
                concrete += ["ira_special", "ira_regular", "ira_variable"]
        else:
            concrete.append(f)
    if q.scope == "business":
        concrete = [f for f in concrete if not f.endswith("_visa") or f.startswith("business_")] or concrete
    q.families = concrete

    q.wants_link = bool(LINK_WORDS.search(t))
    tiered = any(v is not None for v in (q.term_months, q.amount, q.balance, q.model_year))
    rate_word = bool(RATE_WORDS.search(re.sub(r"\b(1st|first) rate\b", " ", t)))   # "1st Rate Checking" is a name
    # "what credit cards do you offer?": a product list (the page outline, else the rate sheet's list)
    q.product_list = len(q.families) > 1 and not rate_word and not tiered and bool(
        re.search(r"\b(what|which)\b.*\b(offer|have|available|choose)\b|\b(types|kinds) of\b", t))
    q.is_rate_question = bool(q.families or q.no_rate) and bool(
        rate_word or tiered or (q.mode != "single" and PRODUCT_WORDS.search(t)) or re.search(r"\bterms\b", t)
        # "what credit cards do you offer?": the product list is the rate sheet's list
        or q.product_list) and not NOT_A_LOOKUP.search(t)
    if q.mode == "single" and q.families and (len(q.families) > 1 or not tiered) and LIST_WORDS.search(t):
        q.mode = "list"
    return q
