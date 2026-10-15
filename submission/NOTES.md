# Dupaco knowledge system: notes

## API keys and external services

**None.** No LLM and no paid API. Everything runs on CPU.

- `pip install -r requirements.txt` installs `fastembed` (ONNX Runtime) and `numpy`.
- On first use, fastembed downloads the **EmbeddingGemma-300m** ONNX model
  (`onnx-community/embeddinggemma-300m-ONNX`, about 1.2 GB) from Hugging Face into
  `submission/.model_cache/`. Set `KS_MODEL_DIR` to use another folder. The model is
  covered by the Gemma Terms of Use.
- If the model can't be downloaded or loaded, the system falls back to BM25 keyword
  search on its own. Rate, organisation, branch and conflict answers don't use
  embeddings, so only free-text questions get worse.
- `data/emb_cache.npz` ships vectors for every chunk of the five snapshots. With it,
  ingesting all five snapshots takes about 20 s, plus the one-time model download (36 s
  in total in a clean Python 3.11 environment). Chunks the cache has never seen are
  embedded within a budget (`KS_EMBED_BUDGET_S`, default 18 min). Any left over when the
  budget runs out are searched with BM25 only, so ingest always stays under 30 minutes.
- `data/sent_cache.npz` (53 MB, built by `dev/build_sent_cache.py`) ships vectors for every
  candidate sentence the extractor can score. Answers then rarely run the model: p95
  latency is about 0.1 s instead of about 2 s. A sentence it lacks is embedded on the fly.

## How "current" is decided

1. **Which snapshot.** With `as_of=None`, the answer comes from the newest snapshot
   (`05_2026-09-28_latest`). With `as_of=<date>`, it comes from the latest snapshot taken
   on or before that date. For example, `2026-05-31` uses the 2026-05-19 crawl. A page's
   printed "effective" date never chooses the snapshot.
2. **Which page, for rates.** Every rate table on every page becomes typed facts, keyed
   by (product, term, balance / loan-amount / model-year band, jumbo, IRA type, rate vs
   APY vs APR). Inside a snapshot, competing pages are ranked by authority:
   - `/rates` and `/business-rates`: 1.0
   - product pages: 0.8 (0.5 for tables off their own topic)
   - alias URLs: 0.6
   - `-lp` landing pages: 0.4
   - PDFs: 0.3
   - calculators, `/homebuying-guidance`, `/mortgage-refi`, `/switch-and-earn`: 0.2
   - blog and news posts: 0.1

   A page the crawler had not re-fetched for more than 14 days (`fetched_at`) counts for
   half. Trusted pages (0.6 or more) beat everything else. Among trusted pages, the
   newest "effective as of" date wins, then authority, then fetch time.
3. **Which tier.** The answer states only the tier that was asked for (term, amount,
   balance, model year), because another tier's number counts as a wrong value. A
   question with no tier gets the headline tier or the list of tiers.
4. **Gone or unpublished.** A product that left the rate sheet is reported as "no longer
   on the current rate sheet", not with its old number. A product with no published rate
   gets an "I don't know, please contact Dupaco" answer with no numbers in it. This covers
   ARMs, construction and lot loans, business loans, student loans, and jumbo mortgages
   over $832,750.
5. **Organisation figures** (branches, members, employees, counties, assets): the newest
   dated source on or before the snapshot date wins. Each press release freezes the
   "About Dupaco" paragraph of its own date, so the site says 19, 20, 22, 23 and 24
   branches at once; the May 5, 2026 release (22 branches) wins. The Summer 2026 Owner's
   Manual "Members: 182,627, as of May 31, 2026" is newer than any release.
6. **Claims that change every year** (board chair, how many times Forbes named Dupaco a
   top credit union): the newest article wins.
7. **Branch hours** exist only in branch-opening press releases. The newest release for
   that branch wins. A branch with no published hours gets an honest "not listed", and
   so do holiday hours. "Which branches are in Dubuque" lists the branch locator's
   entries, so closed branches still in old drop-downs (Asbury, Hy-Vee) are left out.
8. **Everything else** uses dense retrieval plus an extracted answer sentence. Ranking
   favours authoritative pages and newer articles on near-ties, pushes expired offers
   and last cycle's deadlines down, and refuses when the best evidence doesn't cover the
   question. A question that asks for a kind of value (a fee, a limit, a percentage, a
   time, a salary, a count, a SWIFT code) gets an "I don't know" when no well-ranked
   sentence contains one, and so does one about a year or an ordinal ("the 78th annual
   meeting") that the evidence doesn't mention.
9. **List questions** ("what types of loans", "who is on the senior leadership team")
   answer with the matching page's section headings (the product list of a hub page,
   names with titles), not with one sentence. Product lists with rates ("what credit
   cards do you offer") fall back to the rate sheet.
10. **Rate plus fine print.** A rate question that also asks for the catch, eligibility
    or a payout date gets the rate and the fine-print sentence (never a second rate).
    A term the rate sheet doesn't list (a 7-month certificate) gets a refusal, not the
    nearest term's rate.

## conflicts()

`conflicts()` returns about 30 entries. Each entry has competing values, sources (URL,
snapshot, quote), the resolved value (always what `answer()` would say today) and the
reason.

- **Rates** (from the fact table): a stale copy disagrees with the rate sheet inside the
  same snapshot, or the rate sheet changed between snapshots. A rate and its APY or APR
  form one conflict, and the tiers of one product and term form one entry with labelled
  values. Business and IRA copies with identical values join the personal product. Old
  model-year bands are matched to the current band they overlap.
- Traditional vs Roth rows of the same IRA special that disagree. The value the regular
  special also shows wins.
- Expired intro offers (HELOC 1.99%), and discontinued products that an old guide still
  shows (Mortgage Rate Relief).
- Frozen press-release statistics, Forbes recognition count, board chair.
- A sentence on the same page whose amounts changed between snapshots (hardship grant
  $2,000 to $3,500; Rewards Visa minimum cash-back redemption $50 to $25).
- One phone extension printed with two different main numbers (ext. 204).

Tier differences, rate vs APY, and personal vs business products are never reported as
conflicts.

## Code map

| file | role |
|---|---|
| `solution.py` | `build()`; routes each question; caches answers |
| `corpus.py` | markdown cleaning (modals, sidebars, widgets), chunking, article dates |
| `facts.py` | fact table: rate-table parsing, product and tier keys, page authority, resolution |
| `query.py` | rule-based question parser (product, term, amount, model year, intent) |
| `rates.py` | rate answers (right tier only; history goes to `conflict_note`) |
| `orgfacts.py`, `claims.py`, `branches.py` | organisation figures, yearly claims, branch hours, addresses and services |
| `people.py` | "who runs / heads X": staff cards indexed per snapshot, most senior matching title |
| `embedder.py`, `bm25.py`, `retrieval.py`, `extract.py` | retrieval and extractive answers |
| `conflicts.py` | `conflicts()` |

The dev tooling (benchmarks, regression tests) lives in `../dev/`. The embedder was
chosen by benchmark: EmbeddingGemma-300m had dense R@5 0.94 against 0.81 for
bge-small. Every reranker tested lowered R@1, so none is used. A later pilot blended
cross-encoders (ms-marco-MiniLM-L-6/L-12, jina-reranker-v1-tiny, bge-reranker-base) and the
Laya decision model into the extractor's sentence choice (`dev/rerank_pilot.py`, through the
inert `extract.RERANK` hook): every model and weight lost more dev items than it won
(best: 1 won / 3 lost), at 70-670 ms per question, so the hook stays off.

## Results on the public set

`python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl`:

- QA score 0.967. The ceiling without the LLM judge is 0.969, because one item has no
  checkable facts.
- 0 stale-value hits, consistency 1.0, p95 answer latency 0.05 s, ingest about 20 s.

Treat 0.967 as optimistic, because these are the questions the system was tuned on. On
117 reworded and new dev questions, 115 contain the expected facts.

On 51 held-out questions written by agents that never saw the code (`dev/heldout/`), the
first, untuned run scored 0.915 on rates and products and 0.606 on branches, services and
the organisation. Later rounds of fresh held-out questions (`h3`-`h6`, 140 items written blind to
the code, each scored once before any fix it prompted) scored 0.677, 0.690, 0.740 and 0.744
on their first runs. After generic fixes they score 0.918, 0.847, 0.875 and 0.911, so those
later numbers are tuned. `h7` (40 items aimed at the weak categories: as-of dates, stale
traps, newsroom figures) scored 0.578 on its first run and 0.646 after the routing fixes it
exposed. Expect roughly 0.75-0.85 on unseen questions: the high end for
rewordings of public questions, the low end for new topics.

`dev/diagnose.py` runs all dev questions (public, probes, h1-h2; 253 items) and says for
each miss where it was lost: never retrieved, retrieved but outside the sentence pool, in
the pool but another sentence won, values spread over sentences, a rule route, a wrong
refusal, or a stale value. It took the dev score from 0.753 to 0.845. `dev/sweep.py` scores
every dev item at each setting of the extractor's constants and lists the items won and lost.
The fixes they led to:

- **Card fields.** Staff cards, ATM lists and stat tiles are joined into record candidates
  ("Randy Butler; Financial Consultant; Ext. 2917").
- **Sibling rows.** Matching rows and cards are listed together ("wire transfers" gives the
  incoming, domestic and international fees), unless the question names one, by words or
  numbers ("a 5 x 10 box").
- **Two-part questions.** Each half gets its own sentence.
- **Year rules.** Old articles can't answer "last year" or year-total questions, and
  "so far this year" counts lose once the year is over.
- **Refusals.** Fee questions must match the named fee ("advisory fee"). Holiday-hours
  questions, and unknown products ("Coverdell", "a reverse mortgage"), are refused.
- **Rate route guards.** "The most a grant pays" is no rate lookup, and a discontinued
  special names its replacements.
- **Sentence pool.** Sentences come from the top 8 chunks (was 4); with it, sentences dated to
  another year than the one asked lose 0.08, and a deadline already past loses 0.25.
- **Footnotes.** An answer ending in a footnote mark ("... will apply. 4") gets the footnote
  ("Intro rate available only for ... 675-credit score").
- **Addresses.** "Where is X" prefers a sentence with a street address.
- **Routing.** "Which branches have a drive-up" lists the locator's branches with that
  feature; "lobby ... and drive-up" gives both schedules; "how many hours ... training" is no
  opening-hours question; "2025 financial statements" is no 2025-model car loan; "members
  using Bright Track" is no membership count; deadline and cost questions get no bare link.
- **Loan families.** A car-loan question isn't answered by a home-loan sentence.
