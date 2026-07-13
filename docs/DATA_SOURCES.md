# Data sources — options for new data over time

> Where fresh review data can come from once the pipeline needs it: drift detection,
> monitoring, and automated retraining (Milestone 4). Researched 2026-07-13; pick per phase.

## Current dataset (the starting point)

| Aspect | Detail |
|--------|--------|
| Dataset | [`Kerassy/trustpilot-reviews-123k`](https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k) (HuggingFace, MIT) |
| Size | 123,181 reviews · 1,680 companies · 22 categories |
| Collected | one-shot scrape, 19 Dec 2024 → 7 Jan 2025 (UK Trustpilot, English) |
| Columns | `category`, company URL/description, review title, `review`, `stars` |

**Limitation:** it's a static snapshot — it never receives updates, so it cannot by itself
play the role of "new data arriving over time". The three options below fill that gap.

---

## Option 1 — Fresh Trustpilot data *(best realism, most setup)*

Same domain as the served model → true production-style incoming data.

- **Official Trustpilot APIs** — [developer portal](https://developers.trustpilot.com/):
  - [Service Reviews API](https://developers.trustpilot.com/service-reviews-api/) — latest reviews per business unit / language.
  - [Data Solutions API](https://developers.trustpilot.com/data-solutions-api/) — `GET /business-units/{id}/reviews/latest`.
  - **Gotcha:** requires an API key and a Trustpilot business account; free tier is limited.
- **Third-party scrapers** — [Apify Trustpilot actor](https://apify.com/jmmaners/trustpilot-company-reviews/api),
  [Outscraper](https://outscraper.com/trustpilot-reviews-api/). Paid, but no business account needed.
  **Gotcha:** review Trustpilot ToS before scraping for a public repo.
- **DIY re-scrape** — repeat the collection method behind the 123k dataset on a monthly
  schedule; every batch is genuinely new data with real temporal drift.

## Option 2 — Timestamped public datasets *(simulated stream, zero scraping)*

Replay a large historical dataset in time order to emulate arriving data.

| Dataset | Size | Time info | Notes |
|---------|------|-----------|-------|
| [Amazon Reviews 2023 (McAuley Lab)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | 571M reviews | ms-precision Unix timestamps, 1996 → Sep 2023 | Star + text; per-category configs (e.g. `All_Beauty`) keep size sane. **Best drift playground.** ([docs site](https://amazon-reviews-2023.github.io/)) |
| Yelp Open Dataset | ~7M reviews | review dates | Stars + text; same replay trick. |
| Sentiment140 / Capriccio | 1.6M tweets | 3 months of timestamps | Used in drift-detection literature (e.g. [DTU MLOps monitoring module](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/data_drifting/)). |

**Gotcha:** Amazon/Yelp vocabulary ≠ Trustpilot vocabulary — switching sources is a *domain*
shift, not a pure temporal drift. Fine for demonstrating the pipeline; state it in the docs.

## Option 3 — Slice the existing 123k dataset *(zero new deps, fastest)*

- Hold out **categories** or **company buckets** from `Kerassy/trustpilot-reviews-123k`
  and feed them back as "batches arriving over time".
- Simulates **covariate drift** (Electronics reviews read differently from Pets reviews) —
  enough to make Evidently flag a shift and to exercise the retraining trigger.
- No credentials, no scraping, no downloads beyond what `make data` already fetches.

---

## Recommendation

| When | Use | Why |
|------|-----|-----|
| Now → Milestone 4 first pass | **Option 3** (category-batch replay) | Zero deps; proves drift report + retrain trigger end-to-end. |
| Milestone 4 polish / defence demo | **Option 2** (Amazon 2023 monthly slices) | True temporal replay with real timestamps. |
| Only if a live-production story is wanted | **Option 1** (Trustpilot API or scraper) | Real fresh data; needs API key + ToS review first. |

## Links

- Current dataset: <https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k>
- Amazon Reviews 2023: <https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023> · <https://amazon-reviews-2023.github.io/>
- Trustpilot developers: <https://developers.trustpilot.com/> · Service Reviews API: <https://developers.trustpilot.com/service-reviews-api/> · Data Solutions API: <https://developers.trustpilot.com/data-solutions-api/>
- Scrapers: <https://apify.com/jmmaners/trustpilot-company-reviews/api> · <https://outscraper.com/trustpilot-reviews-api/>
- Drift background: <https://skaftenicki.github.io/dtu_mlops/s8_monitoring/data_drifting/>
