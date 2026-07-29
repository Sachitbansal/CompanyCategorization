# optimiseGEO — Brand Categorization Pipeline

## What this is
RAG-style pipeline: input a brand name/website, output a consistent, crisp
category (1 primary + 0-3 secondary tags) used downstream to build GEO
(generative engine optimization) visibility prompts. Consistency across
repeated/similar-brand lookups is the #1 requirement.

## Locked architecture decisions (do not re-litigate without asking)
- 1 mandatory primary tag (drives fast-path lookup) + 0-3 optional secondary
  tags, all from ONE shared canonical tag pool (role is per-company, not
  fixed to the tag).
- Tag dedup guardrail is STRING-LEVEL, not embeddings: normalize (lowercase,
  strip hyphens/whitespace) -> exact match check -> rapidfuzz
  token_set_ratio >= 88 -> only then treat as genuinely new. Runs in plain
  Python AFTER every DeepSeek tag-assignment call, never trust the LLM's
  own "is this new" claim.
- Full category list is passed as plain text in the tag-assignment prompt
  (list stays small) — no vector search over tags.
- CHANGED (final): the summary-generation call returns THREE fields:
  `summary` (50-100 word prose), `keywords`, `confidence`. `summary` is
  EPHEMERAL — used only in-memory as context for (a) keyword generation in
  the same call and (b) the "COMPANY TO CATEGORIZE" block of the
  tag-assignment prompt right after. It is NEVER written to the database
  and NEVER embedded. Once tagging is done for that request, it's
  discarded. The `companies` table has NO summary column.
- `keywords`: industry/sector/product terms only (no geography, no scale
  adjectives, no generic customer-type phrasing). Count is NOT a fixed
  range — approximately 10-12 as a rough center, but a narrow/niche
  business should get fewer (a handful) and a broad/versatile business
  (e.g. a conglomerate spanning several distinct lines of business) should
  get more. The LLM should size the list to genuinely cover the business,
  not pad or truncate to hit a number.
- `keywords` is the ONLY thing embedded (sentence-transformers
  all-MiniLM-L6-v2 -> Chroma) and used for top-k similar-company retrieval.
  Reason for the change: prose-summary embeddings were matching on shared
  sentence structure and boilerplate phrasing ("Indian", "businesses of
  all sizes") rather than actual business similarity — confirmed live when
  Razorpay outranked Flipkart for a Flipkart query. Keyword embeddings
  strip that noise out.
- Storage: SQLite. Tables: companies(id, name, website [normalized: strip
  protocol/www/trailing slash], keywords, primary_tag_id, embedding_id
  [points at the keyword embedding]), categories(id, name, primary_count,
  secondary_count), company_tags(company_id, category_id) as junction
  table for secondaries. NO summary column — summary is never persisted.
- Dashboard/Data page shows: company name, website, assigned primary +
  secondary tags, and the generated keywords. Does NOT show a prose
  summary — intentionally dropped, not needed by the end user.
- Exact-match fast path FIRST, checking BOTH normalized website AND
  normalized name (a name-only lookup must find a website-keyed row and
  vice versa — this was a confirmed bug, fixed).
- Similarity threshold for few-shot retrieval: 0.35 (revised down from an
  initial 0.60 after confirming the original Chroma collection was using
  raw L2 distance while the code assumed cosine — a units-mismatch bug,
  now fixed by creating the collection with
  metadata={"hnsw:space": "cosine"} explicitly). Re-validate this number
  again after the keyword-embedding switch — the score distribution will
  likely shift once boilerplate noise is removed from what's embedded.
- If nothing clears the similarity threshold for top-k retrieval: send NO
  few-shot examples, let the LLM propose fresh.
- Tag-assignment prompt: every secondary tag must be directly evidenced by
  specific content in the company's own (ephemeral) summary/keywords — a
  pool entry existing, or a retrieved similar company carrying a tag, is
  never by itself sufficient justification. (Locked after a confirmed
  leak: Amazon was assigned "Fintech" as secondary purely because Fintech
  existed in the pool, with zero grounding in Amazon's actual summary.)
- Scraping: requests + BeautifulSoup, strict timeout (~3s), User-Agent
  header, extract title/meta description/og:description/JSON-LD
  description/h1 only (not full body text). On failure -> fall back to
  LLM's own knowledge of the brand name.
- If scrape fails AND LLM name-only summary comes back low-confidence ->
  ask the user for more context, loop back into the summary call once
  (not indefinitely).
- DeepSeek calls: temperature=0, structured JSON output, no `is_new`
  self-report field.

## Non-goals / explicitly rejected
- No pgvector/Postgres — SQLite is enough at this scale.
- No tag-embedding vector store — full list fits in-prompt.
- No headless browser (Playwright/Selenium) for JS-heavy sites.
- No live Google search via DeepSeek API.
- No persistence of the prose summary anywhere — generated, used
  in-memory for the current request only, discarded after tagging.

## Working style
Explain what each generated function does in plain terms after writing it.
Prefer Sonnet for boilerplate CRUD/scraper wiring; escalate to Opus for
prompt design and the dedup/canonicalization logic specifically.