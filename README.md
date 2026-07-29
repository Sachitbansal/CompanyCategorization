# Brand Categoriser

Brand categorization pipeline: give it a company name or website, get back
one primary tag + up to three secondary tags, drawn from a single shared,
growing category pool. Built to feed downstream GEO (generative engine
optimization) visibility prompts.

**The one requirement everything else was designed around:** the same
brand, or two genuinely similar brands, must get the same tags every time.
Not "close enough" — the same string, character-for-character. Most of the
decisions below exist because a cheap or obvious approach broke that
guarantee somewhere.

## Pipeline

```
name/website
    │
    ├─ exact-match cache hit? ──────────────► return stored result (no LLM cost)
    │
    ▼
scrape website (title/meta/og/JSON-LD/h1, 3s timeout)
    │  no site, or scrape fails → fall back to brand name alone
    ▼
DeepSeek: summary + keywords + confidence   (temperature=0)
    │  confidence=low → pause, ask user for a one-line description, retry once
    ▼
embed keywords → Chroma cosine search → top-k similar companies (≥ threshold)
    │  nothing clears the bar → send zero few-shot examples, let it propose fresh
    ▼
DeepSeek: assign 1 primary + 0-3 secondary tags (sees pool + few-shot examples)
    │
    ▼
string-level dedup guardrail (normalize → exact match → rapidfuzz ≥ 88)
    │  this, not the LLM, decides what's actually a new tag
    ▼
write company + tags + embedding
```

## Why it's built this way

**Dedup is plain Python string matching, not embeddings, and never trusts
the LLM's own opinion.** The tagging prompt has no `is_new` field at all —
after every call, every proposed tag gets normalized (lowercase, strip
hyphens, strip filler words like "platform"/"services") and checked against
the existing pool: exact match first, then `rapidfuzz.token_set_ratio ≥ 88`.
Only if both miss does it become a genuinely new category. This is the
actual mechanism behind the consistency guarantee — everything upstream
(prompts, retrieval) exists to make the LLM's *first guess* good, but this
step is what makes wrong guesses non-catastrophic.

**The prose summary is ephemeral — generated, used for one request, thrown
away.** Only `keywords` gets stored and embedded. This wasn't the original
design; it came from a live bug: prose-summary embeddings matched on shared
sentence boilerplate ("Indian", "businesses of all sizes") rather than
actual business similarity, and Razorpay literally outranked Flipkart for a
Flipkart query. Keywords strip that noise out because there's no sentence
structure left to match on.

**Tags aren't scoped to primary or secondary — one shared pool, role is
per-company.** "Fashion & Lifestyle" is primary for a clothing brand and
secondary for a marketplace that sells it alongside everything else. This
keeps the pool small and reusable instead of doubling it.

**Every secondary tag needs evidence in the company's own keywords/summary
— being in the pool, or appearing on a similar retrieved company, is never
enough on its own.** Locked in after a real failure: Amazon got tagged
"Fintech" as secondary purely because Fintech existed in the pool, with
nothing in Amazon's actual description supporting it. The tagging prompt
now states this rule twice — once while it's reasoning, once right before
it answers — because whatever it read last (the category pool) had the
strongest pull.

**The similarity threshold has been recalibrated four times, on purpose,
against measured data each time** — not guessed: 0.60 → 0.35 (fixed a real
bug: the Chroma collection was silently using squared-L2 distance while the
code assumed cosine) → 0.65 (after switching from summaries to keyword
embeddings, which shifted the whole score range up) → 0.55 (0.65 was
rejecting genuine matches like HP/Dell at 0.584) → back to 0.60 (0.55 let
marginal cross-industry pairs through — Zomato pulled in Flipkart and
Alibaba as "similar" and got mistagged E-commerce as a result). The
takeaway that stuck: a missing few-shot example is cheaper than a wrong
one — with none, the model falls back to reasoning from the company's own
keywords, which is the safer failure mode.

**Exact-match cache checks both website and name, not just one.** A
name-only lookup for "amazon" used to miss the existing website-keyed
"amazon.com" row and silently create a duplicate. Fixed by checking both
keys and treating a hit on either as the same company.

**Scraping is deliberately shallow.** requests + BeautifulSoup, 3s timeout,
pulls only title/meta description/og:description/JSON-LD description/h1 —
never full body text. No headless browser, no JS rendering. If it fails or
the page has nothing usable, the pipeline falls back to the LLM's own
knowledge of the brand name rather than trying harder to scrape.

**If nothing gives the model enough to work with, it asks instead of
guessing.** A low-confidence summary — whether from a failed scrape and an
unfamiliar name, or a scrape that succeeded but returned nothing
substantive — pauses the pipeline and asks the user for a one-line
description. That description gets rewritten by DeepSeek into the same
clean, consistent shape as every other summary before continuing, so it
doesn't stick out downstream.

## What it deliberately doesn't do

- No vector DB for tags — the category pool stays small enough to pass as
  plain text in the prompt, so there's nothing to index.
- No Postgres/pgvector — SQLite is enough at this scale.
- No headless browser — a 3s timeout + name-only fallback covers this well
  enough without the operational cost of running Chromium.
- No live web search to resolve "brand name → website" — only the URL
  given is scraped, nothing else is looked up.

## Stack

- **Backend** (`backend/`) — FastAPI. WebSocket endpoint streams pipeline
  progress live (each step, plus which companies matched during retrieval
  and their similarity scores); REST endpoints back the Data page.
  SQLite for companies/categories/tags, Chroma for keyword embeddings
  (`all-MiniLM-L6-v2`), DeepSeek (OpenAI-compatible API) for the two LLM
  calls.
- **Frontend** (`frontend/`) — plain HTML/CSS/JS, no framework, no build
  step. Two pages: categorize (live log + result) and Data (every company,
  searchable, deletable).
- **Docker** — `docker compose up -d` runs both: backend on `8006`,
  frontend on `8009`. `backend/data/` is a mounted volume so the DB and
  vector store survive rebuilds.

## Running it

```bash
cp backend/.env.example backend/.env   # add your DeepSeek key as KEY=...
docker compose up -d --build
```

Then open `http://localhost:8009`.

## Layout

```
backend/
  main.py           orchestration — run_pipeline() drives every stage
  server.py          FastAPI app — WebSocket + REST wrapper around main.py
  db.py               SQLite schema + CRUD
  scraper.py          website metadata extraction
  embed.py             keyword embedding + Chroma retrieval
  dedup.py              string-level tag canonicalization guardrail
  prompts/               the two DeepSeek prompt templates + JSON schemas
frontend/
  index.html / app.js    categorize page + live pipeline log
  data.html / data.js     browse / search / delete companies
  style.css                 shared design tokens
```
