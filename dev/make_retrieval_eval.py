"""
Builds dev/retrieval_eval.jsonl: retrieval test queries with relevance labels.

A chunk is relevant to a query when
  * it is live in the query's target snapshot (latest snapshot taken on/before as_of,
    else the newest one),
  * its URL path is one of `rel` (empty `rel` = any page), and
  * its title + section + text contain every token of at least one `ans` alternative
    (numbers compared by value, as the official scorer does).

Queries cover the public practice questions, re-worded variants of them (the hidden
set re-words questions), and new questions taken from the snapshot survey.
"""
import json
from pathlib import Path

RATES = ["/rates"]
SAV = ["/personal/savings-account", "/personal/savings/holiday-club-account"]
MORT = ["/personal/loans/home-loans", "/learn/master-your-money/homeownership", "/personal/loans/home-improvement-loan"]
AUTO = ["/personal/loans/auto-motorcycle-loans", "/personal/loans/auto-loans"]
CARDS = ["/personal/credit-cards"]
BCARDS = ["/business/credit-cards"]
RET = ["/personal/retirement"]
CHK = ["/personal/checking-accounts", "/checking"]
HE = ["/personal/loans/home-equity"]
PL = ["/personal/loans/personal-loans", "/personal-loans", "/loans"]
REC = ["/personal/loans/recreational-loans"]
JOIN = ["/about/join", "/membership"]
ANY: list = []

Q = []


def q(qid, text, rel, ans, as_of=None, group=None):
    Q.append({"id": qid, "q": text, "as_of": as_of, "rel": rel, "ans": ans, "group": group or qid.split(".")[0]})


# ---- public practice questions + re-wordings -------------------------------------------
q("p001", "What is the APY on Dupaco's 17-month special certificate?", RATES + SAV, [["17 Month", "4.10%"]])
q("p001.a", "How much does the 17 month certificate special pay?", RATES + SAV, [["17 Month", "4.10%"]])
q("p001.b", "What's the yield on your 17-mo CD special?", RATES + SAV, [["17 Month", "4.10%"]])
q("p009", "Can you tell me the interest rate for a 30-year mortgage?", RATES + MORT, [["30-Year", "7.13"]])
q("p009.a", "What is the current 30 year fixed home loan rate?", RATES + MORT, [["30-Year", "7.13"]])
q("p009.b", "thirty-year fixed rate mortgage interest", RATES + MORT, [["30-Year", "7.13"]])
q("p011", "What are the mortgage rates right now?", RATES + MORT, [["7.13"]])
q("p016", "What's the APR for a 60-month auto loan on a 2025 model-year vehicle?", RATES + AUTO, [["60 Months", "5.39%"]])
q("p016.a", "Car loan rate for a 2025 vehicle financed over 60 months?", RATES + AUTO, [["60 Months", "5.39%"]])
q("p016.b", "If I finance a 2025 car for five years, what APR would I get?", RATES + AUTO, [["60 Months", "5.39%"]])
q("p017", "What's the APR for an 84-month auto loan on a 2024 vehicle?", RATES + AUTO, [["84 Months", "5.59%"]])
q("p017.a", "84 month car loan rate for a 2024 model year", RATES + AUTO, [["84 Months", "5.59%"]])
CARD_ANS = [["Rewards", "15.50%"], ["Platinum", "12.50%"], ["Business Platinum", "15.50%"]]
q("p034", "What credit cards do you offer and what are their APRs?", CARDS + RATES + BCARDS, CARD_ANS)
q("p034.a", "Which Visa cards does Dupaco have and what do they charge?", CARDS + RATES + BCARDS, CARD_ANS)
q("p040", "I would like to know the rates for the credit cards", CARDS + RATES + BCARDS, CARD_ANS)
q("p035", "What is the standard purchase APR on the Dupaco Platinum Visa?", CARDS + RATES, [["Platinum", "12.50%"]])
q("p035.a", "Platinum Visa interest rate, not the intro rate", CARDS + RATES, [["Platinum", "12.50%"]])
q("p045", "What dividend rate do I earn on Share Savings with a balance between $10,000 and $24,999?", RATES + SAV, [["Share Savings", "0.15%"]])
q("p045.a", "Share savings rate for a $15,000 balance", RATES + SAV, [["Share Savings", "0.15%"]])
q("p046", "What is the highest Share Savings APY?", RATES + SAV, [["Share Savings", "0.20%"]])
q("p049", "Do any Dupaco savings tiers pay 0.35% APY?", RATES + SAV, [["Share Savings", "0.20%"]])
BRIGHT = RATES + SAV + ["/savings-account/online-savings-account"]
q("p050", "What is the rate for the Bright Savings account?", BRIGHT, [["Bright Savings", "2.25%"]])
q("p050.a", "Bright Savings APY?", BRIGHT, [["Bright Savings", "2.25%"]])
q("p050.b", "What does your online-only savings account pay?", BRIGHT, [["Bright Savings", "2.25%"]])
q("p052", "Provide current checking account interest rates", RATES + CHK, [["1st Rate", "2.50%"]])
q("p052.a", "Does 1st Rate Checking earn interest, and how much?", RATES + CHK, [["1st Rate", "2.50%"]])
q("p028", "What is the APR on a 36-month personal loan?", RATES + PL, [["36 Months", "10.49%"]])
q("p028.a", "personal loan rate for 3 years", RATES + PL, [["36 Months", "10.49%"]])
q("p006", "What was the rate on the 10-month fixed jumbo special certificate as of May 2026?", RATES + SAV, [["10 Month", "Jumbo", "3.90%"]], as_of="2026-05-31")
q("p005", "Get me the rate for the certificate of accounts.", RATES + SAV, [["12 Month", "3.40%"]])
q("p005.a", "What are your CD rates?", RATES + SAV, [["12 Month", "3.40%"]])
q("p005.b", "List all certificate terms and their rates", RATES + SAV, [["12 Month", "3.40%"]])
q("p054", "What is Dupaco's routing number?", ANY, [["273974581"]])
q("p054.a", "ABA routing number for direct deposit", ANY, [["273974581"]])
q("p054.b", "routing # for Dupaco Community Credit Union", ANY, [["273974581"]])
LOBBY = [["Lobby", "8:30"]]
q("p064", "What are Dupaco's general lobby hours?", ANY, LOBBY)
q("p064.a", "What time do branch lobbies open?", ANY, LOBBY)
q("p065", "What are your lobby hours?", ANY, LOBBY)
q("p066", "What are the drive-up hours?", ANY, [["Drive-up", "8 a.m."]])
q("p066.a", "When is the drive-thru open?", ANY, [["Drive-up", "8 a.m."]])
q("p104", "Do all Dupaco branches keep the same lobby hours?", ANY, [["Lobby", "12:30"], ["Lobby", "9 a.m."]])
q("p068", "What is the address of the Galena Branch?", ANY, [["11375 Oldenburg"]])
q("p068.a", "Where is the Galena, Illinois location?", ANY, [["11375 Oldenburg"]])
q("p073", "How many branches does Dupaco have?", ANY, [["22 branch"]])
q("p073.a", "How many locations does Dupaco operate?", ANY, [["22 branch"]])
q("p073.b", "number of Dupaco branch offices", ANY, [["22 branch"]])
HYVEE = [["Hy-Vee", "March 31, 2026"]]
q("p074", "When does the Hy-Vee branch close?", ANY, HYVEE)
q("p074.a", "Is the Hy-Vee location closing?", ANY, HYVEE)
q("p075", "Is the Hy-Vee branch still open today (October 2026)?", ANY, HYVEE)
q("p076", "Where can Hy-Vee branch members go for full-service banking?", ANY, [["Key West", "Sycamore"]])
q("p087", "What recognition has Dupaco received from Forbes?", ANY, [["Forbes", "fifth"]])
q("p087.a", "Has Forbes named Dupaco a top credit union?", ANY, [["Forbes", "fifth"]])
q("p088", "How many Dupaco employees used Volunteer Time Off last year?", ANY, [["294"], ["256"]])
q("p088.a", "How many staff took paid volunteer time?", ANY, [["294"], ["256"]])
q("p008", "What link should I use to see Dupaco's CD / certificate rates?", SAV + RATES, [["Certificate"]])
q("p080", "What's the URL for Dupaco's branch and ATM locator?", ["/contact"], [[]])
q("p080.a", "Where can I find an ATM near me?", ["/contact"], [[]])
q("p082", "How do I join Dupaco online?", JOIN, [["join.dupaco.com"]])
q("p082.a", "How can I become a member online?", JOIN, [["join.dupaco.com"]])
q("p081", "Who is eligible to become a Dupaco member?", JOIN, [["Iowa"]])
q("p081.a", "Can I join if I live in Wisconsin?", JOIN, [["Iowa"]])
q("p031", "What is Dupaco's Home$tart program?", ANY, [["Home$tart", "15,000"]])
q("p094", "Can I add my Dupaco debit and credit cards to a mobile wallet?", ANY, [["Apple Pay", "Samsung Pay"]])
q("p094.a", "Does Dupaco support Apple Pay?", ANY, [["Apple Pay", "Samsung Pay"]])
q("p101", "I'm a student going to a new university - do you have any loan options with flexibility?", ["/personal/loans/student-loans"], [[]])
q("p101.a", "Do you offer student loans?", ["/personal/loans/student-loans"], [[]])

# ---- new questions from the snapshot survey ----------------------------------------
q("n01", "What is the HELOC rate?", RATES + HE, [["6.25%"]])
q("n01.a", "home equity line of credit APR", RATES + HE, [["6.25%"]])
q("n02", "What is the rate on a 12-month certificate?", RATES + SAV, [["12 Month", "3.40%"]])
q("n03", "What was the 12-month certificate rate in May 2026?", RATES + SAV, [["12 Month", "2.90%"]], as_of="2026-05-20")
q("n04", "What did 1st Rate Checking pay in mid May 2026?", RATES + CHK, [["1st Rate", "3.00%"]], as_of="2026-05-15")
q("n05", "Home equity loan rate for 120 months", RATES + HE, [["120", "7.24%"]])
q("n06", "Home improvement loan rate for borrowing $30,000", RATES, [["8.75%"]])
q("n07", "Boat loan APR for $30,000 over 72 months", RATES + REC, [["72 Months", "7.34%"]])
q("n08", "ATV loan rate for a 2025 model", RATES + REC, [["8.34%"]])
q("n09", "What is the Rewards Visa APR?", RATES + CARDS, [["Rewards", "15.50%"]])
q("n10", "Business Platinum Visa interest rate", BCARDS, [["15.50%"]])
q("n11", "High-Yield Savings rate for a $600,000 balance", RATES + SAV, [["High-Yield", "2.25%"]])
q("n12", "Top tier rate on Investor's Choice Money Market", RATES + SAV, [["0.40%"]])
q("n13", "HSA interest rate", RATES + SAV, [["Health Savings", "0.35%"]])
q("n14", "Launchpad savings rate", RATES + RET, [["Launchpad", "3.25%"]])
q("n15", "Achieve Money Market introductory rate", ["/earn-more", "/rates"] + SAV, [["3.66%"]])
q("n16", "Traditional IRA 36-month certificate rate", RATES + RET, [["36 Months", "2.10%"]])
q("n17", "Who is Dupaco's CEO?", ANY, [["Joe Hearn"]])
q("n18", "How many members does Dupaco have?", ANY, [["182,627"], ["179,000"]])
q("n19", "How much can the hardship grant give?", ["/hardship-grant"], [["3,500"]])
q("n20", "When was Dupaco founded?", ANY, [["1948"]])
q("n21", "What is the overdraft fee?", ANY, [["overdraft", "$30"]])
q("n22", "Incoming wire transfer fee", ANY, [["Incoming wire", "$10"]])
q("n23", "When did the Asbury branch close?", ANY, [["Asbury", "Oct. 1, 2024"]])
q("n24", "Cedar Falls branch address", ANY, [["126 Brandilynn"]])
q("n27", "Gift of Innovation grant application deadline", ["/gift-of-innovation-grant"], [["Nov. 12, 2026"]])
q("n28", "How do I activate my debit card?", ANY, [["888-691-8661"]])
q("n29", "How much student loan debt can I refinance?", ANY, [["125,000"]])
q("n30", "What is a Mini-Max loan?", ANY, [["Mini-Max"]])
q("n31", "What credit score do I need for the best HELOC rate?", RATES + HE, [["720"]])
q("n32", "What is the jumbo mortgage loan limit?", ANY, [["832,750"]])
q("n33", "How many counties does Dupaco serve?", ANY, [["118 counties"]])
q("n34", "When is the scholarship application period?", ["/scholarships", "/community"], [["Feb. 24, 2027"]])
q("n35", "17-month certificate special rate as of July 2026", RATES + SAV, [["17 Month", "3.85%"]], as_of="2026-07-20")
q("n36", "30-year mortgage rate as of May 31, 2026", RATES + MORT, [["30-Year", "6.250%"]], as_of="2026-05-31")
q("n37", "HELOC introductory rate in June 2026", RATES + HE, [["1.99%"]], as_of="2026-06-30")
q("n38", "60-month auto loan rate in August 2026", RATES + AUTO, [["60 Months", "5.19%"]], as_of="2026-08-01")
q("n39", "CHECK300 promo code offer", ANY, [["CHECK300"]])
q("n40", "How do I refinance my car loan to Dupaco and get cash back?", ["/switch-and-earn"], [[]])
q("n41", "What is the late payment fee on the Platinum Visa?", ANY, [["late payment", "$15"]])
q("n42", "What is the foreign transaction fee on Dupaco credit cards?", ANY, [["foreign transaction", "1.00%"]])
q("n45", "What's the minimum deposit to open a certificate?", RATES + SAV, [["$1,000"]])
q("n46", "Who do I call about mortgage rates?", ANY, [["ext. 204"]])
q("n47", "What is Dupaco's main phone number?", ["/contact"], [["800-373-7600"]])
q("n48", "How do I reset my Shine online banking password?", ["/connect/online-and-mobile-banking/shine-login-help"], [[]])

out = Path(__file__).with_name("retrieval_eval.jsonl")
out.write_text("\n".join(json.dumps(x) for x in Q) + "\n", encoding="utf-8")
print(f"wrote {len(Q)} queries to {out}")
