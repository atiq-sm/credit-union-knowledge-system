"""
End-to-end check of question parser + fact-table answers, graded with the official
scorer (harness/scoring.py).

  1. public eval items the parser routes to the rate path
  2. re-worded / new rate questions (the hidden set re-words questions)

    .venv/Scripts/python dev/test_rates.py [-v]
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from facts import build_fact_table  # noqa: E402
from query import parse  # noqa: E402
from rates import answer_rate  # noqa: E402
from scoring import score_item  # noqa: E402

VERBOSE = "-v" in sys.argv
chunks, dates = common.corpus()
ft = build_fact_table(chunks, dates)


def run(item):
    q = parse(item["question"], item.get("as_of"))
    a = answer_rate(q, ft) if q.is_rate_question else None
    if a is None:
        return q, None, None
    ans = SimpleNamespace(text=a["text"], sources=[SimpleNamespace(url=u) for u in a["sources"]],
                          confident=a["confident"])
    return q, a, score_item(item, ans)


# ---------------------------------------------------------------- 1. public eval
items = [json.loads(l) for l in (common.ROOT / "eval" / "public_eval.jsonl").read_text().splitlines() if l.strip()]
print("== public eval items routed to the rate path")
tot = n = 0
for it in items:
    q, a, r = run(it)
    if a is None:
        continue
    n += 1
    tot += r["score"]
    flag = "ok " if r["score"] >= 0.85 else "LOW"
    print(f"{flag} {r['score']:.2f} {it['id']} [{it['category']}] {it['question'][:70]}")
    if VERBOSE or r["score"] < 0.85:
        print(f"      fam={q.families} term={q.term_months} amt={q.amount} bal={q.balance} my={q.model_year} "
              f"mode={q.mode} as_of={q.as_of}\n      -> {a['text'][:400]}\n      missed={r['missed']} traps={r['trap_hits']} src={r['source_hit']}")
print(f"routed {n}/{len(items)}, mean score {tot / max(n, 1):.3f}")
unrouted = [it["id"] for it in items if run(it)[1] is None]
print("not routed (left for retrieval):", unrouted)

# ------------------------------------------------- 2. re-worded and new questions
CASES = [
    # (question, as_of, must_include, must_not_include)
    ("How much does the 17 month certificate special pay?", None, ["4.10%"], ["4.20%", "3.95%"]),
    ("What's the yield on your 17-mo CD special?", None, ["4.10%"], ["4.20%"]),
    ("17-month jumbo certificate APY", None, ["4.20%"], ["4.10%"]),
    ("What's the 10-month certificate special APY?", None, ["3.90%"], ["4.00%", "3.80%"]),
    ("10 month jumbo CD rate in May 2026", None, ["3.90%"], ["4.00%"]),
    ("What is the current 30 year fixed home loan rate?", None, ["7.13", "7.25"], ["6.037%", "6.375%"]),
    ("thirty-year fixed rate mortgage interest", None, ["7.13"], ["6.037%"]),
    ("What would the rate be on a $250,000 30-year mortgage?", None, ["7.25%", "7.32%"], ["7.13%", "7.29%"]),
    ("15-year mortgage rate", None, ["6.75%"], ["5.50%", "6.00%"]),
    ("What was the 30-year mortgage rate on May 31, 2026?", None, ["6.25%"], ["7.13%"]),
    ("Car loan rate for a 2025 vehicle financed over 60 months?", None, ["5.39%"], ["5.34%", "5.19%"]),
    ("If I finance a 2025 car for five years, what APR would I get?", None, ["5.39%"], ["5.34%"]),
    ("84 month car loan rate for a 2024 model year", None, ["5.59%"], ["5.54%"]),
    ("auto loan rate for a 2019 truck, 72 months", None, ["6.49%"], ["6.44%", "6.29%"]),
    ("What was the 60-month auto loan rate for a 2025 car in August 2026?", None, ["5.19%"], ["5.39%"]),
    ("Share savings rate for a $15,000 balance", None, ["0.15%"], ["0.20%", "0.10%", "0.30%"]),
    ("What is the highest Share Savings APY?", None, ["0.20%"], ["0.35%"]),
    ("Do any savings tiers pay 0.35% APY?", None, ["0.20%"], ["up to 0.35"]),
    ("Bright Savings APY?", None, ["2.25%"], ["0.35%"]),
    ("What does your online-only savings account pay?", None, ["2.25%"], []),
    ("Does 1st Rate Checking earn interest, and how much?", None, ["2.50%"], ["3.00%"]),
    ("What did 1st Rate Checking pay in mid May 2026?", "2026-05-15", ["3.00%"], ["2.50%"]),
    ("personal loan rate for 3 years", None, ["10.49%"], ["9.99%", "10.99%"]),
    ("What is the HELOC rate?", None, ["6.25%"], ["1.99%"]),
    ("home equity line of credit APR", None, ["6.25%"], []),
    ("What was the HELOC intro rate in June 2026?", None, ["1.99%"], []),
    ("Home equity loan rate for 10 years", None, ["7.24%"], ["6.74%"]),
    ("Home improvement loan rate for borrowing $30,000", None, ["8.75%"], ["8.50%"]),
    ("Boat loan APR for $30,000 over 72 months", None, ["7.34%"], ["7.59%"]),
    ("ATV loan rate for a 2025 model", None, ["8.34%"], ["8.89%"]),
    ("What is the Rewards Visa APR?", None, ["15.5", "21"], ["12.5"]),
    ("Platinum Visa interest rate, not the intro rate", None, ["12.5"], ["9.9%"]),
    ("What's the balance transfer intro rate on the Platinum card?", None, ["2.90%"], []),
    ("Business Platinum Visa interest rate", None, ["15.50%"], ["12.50%"]),
    ("What credit cards do you offer and what are their APRs?", None, ["12.5", "15.5", "20.5"], ["9.9%"]),
    ("High-Yield Savings rate for a $600,000 balance", None, ["2.25%"], ["2.45%", "2.05%"]),
    ("Top tier rate on Investor's Choice Money Market", None, ["0.40%"], []),
    ("HSA interest rate for $30,000", None, ["0.35%"], ["0.30%"]),
    ("Launchpad savings rate", None, ["3.25%"], []),
    ("Achieve Money Market introductory rate", None, ["3.66%"], ["3.25%"]),
    ("Traditional IRA 36-month certificate rate", None, ["2.10%"], ["2.70%"]),
    ("What is your rate on a 7/1 adjustable-rate mortgage?", None, [], []),
    ("What interest rate do you charge on a construction loan?", None, [], []),
    ("What's the rate for a business equipment loan?", None, [], []),
]
print("\n== re-worded / new rate questions")
ok = 0
for qtext, as_of, inc, exc in CASES:
    item = {"id": qtext[:20], "question": qtext, "as_of": as_of, "must_include": inc, "must_not_include": exc,
            "category": "should_refuse" if not inc else "x"}
    q, a, r = run(item)
    good = r is not None and r["correctness"] >= 0.999
    ok += good
    if VERBOSE or not good:
        txt = a["text"][:300] if a else "(not routed)"
        print(f"{'ok ' if good else 'BAD'} {qtext}\n      fam={q.families} term={q.term_months} amt={q.amount} "
              f"bal={q.balance} my={q.model_year} mode={q.mode} as_of={q.as_of} no_rate={q.no_rate}\n      -> {txt}"
              + (f"\n      missed={r['missed']} traps={r['trap_hits']}" if r else ""))
print(f"{ok}/{len(CASES)} correct")
