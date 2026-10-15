"""
Checks the fact table against values verified by hand in the snapshot survey,
including the traps (stale calculators, old snapshots, tier collisions).

    .venv/Scripts/python dev/test_facts.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from facts import build_fact_table, fmt_pct, page_path  # noqa: E402

chunks, dates = common.corpus()
ft = build_fact_table(chunks, dates)
CUR = ft.snapshot_for(None)

# (description, family, kwargs, metric, expected text, expected source path or None)
CASES = [
    ("60-mo auto, 2025 vehicle (not the stale 5.34% calculator)", "auto_loan",
     dict(snapshot=CUR, term_months=60, model_year=2025), "apr", "5.39%", "/rates"),
    ("84-mo auto, 2024 vehicle", "auto_loan", dict(snapshot=CUR, term_months=84, model_year=2024), "apr", "5.59%", "/rates"),
    ("60-mo auto 2025 as of Aug 2026 (snapshot 03)", "auto_loan",
     dict(snapshot=ft.snapshot_for("2026-08-01"), term_months=60, model_year=2025), "apr", "5.19%", "/rates"),
    ("30-yr purchase <=$100k rate", "mortgage_purchase", dict(snapshot=CUR, term_months=360, amount=100000), "rate", "7.13%", "/rates"),
    ("30-yr purchase <=$100k APR", "mortgage_purchase", dict(snapshot=CUR, term_months=360, amount=100000), "apr", "7.27%", "/rates"),
    ("30-yr purchase rate as of 2026-05-31", "mortgage_purchase",
     dict(snapshot=ft.snapshot_for("2026-05-31"), term_months=360, amount=100000), "rate", "6.25%", "/rates"),
    ("15-yr purchase rate", "mortgage_purchase", dict(snapshot=CUR, term_months=180), "rate", "6.75%", "/rates"),
    ("17-mo special APY (non-jumbo)", "certificate_special", dict(snapshot=CUR, term_months=17, jumbo=False), "apy", "4.10%", "/rates"),
    ("17-mo jumbo special APY", "certificate_special", dict(snapshot=CUR, term_months=17, jumbo=True), "apy", "4.20%", "/rates"),
    ("10-mo jumbo special APY as of May 2026", "certificate_special",
     dict(snapshot=ft.snapshot_for("2026-05-31"), term_months=10, jumbo=True), "apy", "3.90%", "/rates"),
    ("10-mo jumbo special APY now", "certificate_special", dict(snapshot=CUR, term_months=10, jumbo=True), "apy", "4.00%", "/rates"),
    ("12-mo certificate APY now", "certificate_regular", dict(snapshot=CUR, term_months=12), "apy", "3.40%", "/rates"),
    ("12-mo certificate APY May 2026", "certificate_regular",
     dict(snapshot=ft.snapshot_for("2026-05-20"), term_months=12), "apy", "2.90%", "/rates"),
    ("Share Savings $15,000 dividend rate", "share_savings", dict(snapshot=CUR, balance=15000), "rate", "0.15%", "/rates"),
    ("Share Savings $25,000+ APY", "share_savings", dict(snapshot=CUR, balance=30000), "apy", "0.20%", "/rates"),
    ("Bright Savings APY", "bright_savings", dict(snapshot=CUR), "apy", "2.25%", "/rates"),
    ("1st Rate Checking $10k APY now", "first_rate_checking", dict(snapshot=CUR, balance=10000), "apy", "2.50%", "/rates"),
    ("1st Rate Checking $10k APY mid-May", "first_rate_checking",
     dict(snapshot=ft.snapshot_for("2026-05-15"), balance=10000), "apy", "3.00%", "/rates"),
    ("36-mo personal loan APR", "personal_loan", dict(snapshot=CUR, term_months=36), "apr", "10.49%", "/rates"),
    ("HELOC APR", "heloc", dict(snapshot=CUR), "apr", "6.25%", "/rates"),
    ("HELOC intro as of June 2026", "heloc", dict(snapshot=ft.snapshot_for("2026-06-30")), "intro_apr", "1.99%", "/rates"),
    ("Platinum Visa standard APR", "platinum_visa", dict(snapshot=CUR), "apr", "12.50%–19.50%", "/rates"),
    ("Platinum Visa intro APR", "platinum_visa", dict(snapshot=CUR), "intro_apr", "2.90%", "/rates"),
    ("Rewards Visa APR", "rewards_visa", dict(snapshot=CUR), "apr", "15.50%–21.00%", "/rates"),
    ("Business Platinum Visa APR (only on /business/credit-cards in 05)", "business_platinum_visa",
     dict(snapshot=CUR, scope="business"), "apr", "15.50%", "/business/credit-cards"),
    ("Home equity loan 120-mo", "home_equity_loan", dict(snapshot=CUR, term_months=120), "apr", "7.24%", "/rates"),
    ("Home improvement $30,000", "home_improvement_loan", dict(snapshot=CUR, amount=30000), "apr", "8.75%", "/rates"),
    ("Boat $30,000 72-mo", "boat_rv_loan", dict(snapshot=CUR, amount=30000, term_months=72), "apr", "7.34%", "/rates"),
    ("ATV 2025 model 60-mo", "atv_loan", dict(snapshot=CUR, model_year=2025, term_months=60), "apr", "8.34%", "/rates"),
    ("HSA $25,000+", "hsa", dict(snapshot=CUR, balance=30000), "apy", "0.35%", "/rates"),
    ("Investor's Choice $100k+", "money_market_investors_choice", dict(snapshot=CUR, balance=150000), "apy", "0.40%", "/rates"),
    ("High-Yield $600k", "high_yield_savings", dict(snapshot=CUR, balance=600000), "apy", "2.25%", "/rates"),
    ("Achieve intro APY", "money_market_achieve", dict(snapshot=CUR, balance=50000), "intro_apy", "3.66%", "/earn-more"),
    ("Traditional IRA 36-mo", "ira_regular", dict(snapshot=CUR, term_months=36, ira="traditional"), "apy", "2.10%", "/rates"),
    ("Business 17-mo special in snapshot 04 (product page newer than /business-rates)", "certificate_special",
     dict(snapshot=3, scope="business", term_months=17, jumbo=False), "apy", "3.95%", "/business/savings-accounts"),
]

fails = 0
for desc, fam, kw, metric, want, src in CASES:
    hits = [(k, o) for k, o in ft.lookup(fam, metrics=[metric], **kw)]
    got = hits[0][1] if hits else None
    ok = got is not None and got.text == want and (src is None or page_path(got.url) == src)
    fails += not ok
    detail = f"{got.text} @ {page_path(got.url)}" if got else "no match"
    extra = f"  (+{len(hits) - 1} more)" if len(hits) > 1 else ""
    print(f"{'ok ' if ok else 'FAIL'} {desc}: {detail}{extra}" + ("" if ok else f"   expected {want} @ {src}"))
print(f"\n{len(CASES) - fails}/{len(CASES)} passed")
sys.exit(1 if fails else 0)
