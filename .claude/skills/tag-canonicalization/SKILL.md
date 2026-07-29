---
name: similarity-retrieval
description: Use whenever writing or modifying embed.py, the Chroma collection setup, or the top-k similar-company retrieval step feeding few-shot examples into the tagging prompt. Covers what gets embedded, the distance-space bug fix, and the current similarity threshold.
---

# Similarity retrieval for few-shot tagging context

## What gets embedded — and what does NOT
- Embed ONLY `keywords`: industry/sector/product terms per company, sized
  to the business (roughly 10-12 as a loose center, fewer for a narrow/
  niche business, more for a broad/versatile one — never padded or
  truncated just to hit a number).
- Do NOT embed the prose `summary`. The summary is generated as an
  EPHEMERAL, in-memory intermediate — used only as context for keyword
  generation and for the tagging prompt's "COMPANY TO CATEGORIZE" block
  within the same request — and is discarded afterward. It is never
  written to the database and never sent to Chroma.
- Confirmed bug history for why prose summaries were dropped from
  embedding entirely: they embed on shared sentence structure and
  boilerplate phrasing ("Indian", "businesses of all sizes", "competes in
  the X space") rather than actual business similarity — this caused
  Razorpay to outrank Flipkart for a Flipkart query, and Amazon/Flipkart
  (same industry) to score lower than Amazon/Razorpay (different
  industries).

## Distance space — do not repeat this bug
Chroma collections created without an explicit metric silently default to
squared-L2 distance, NOT cosine distance. `similarity = 1 - distance` is
ONLY valid math when `distance` is a cosine distance (range 0-2) — applied
to raw L2 numbers it produces meaningless results (confirmed: Flipkart vs
eBay came out as -0.58, despite being genuinely similar businesses).

Always create the collection explicitly:
```python
collection = client.get_or_create_collection(
    name="company_keywords",
    metadata={"hnsw:space": "cosine"},
)
```
With this set, Chroma's returned `distance` is a real cosine distance, and
`similarity = 1 - distance` is then correct.

## Similarity threshold
Currently 0.35 (tunable, per CLAUDE.md). This was recalibrated once
already, from an initial 0.60 that turned out to reject every real
company-pair comparison under this model. Re-validate this number
specifically after the keyword-embedding switch and again after moving to
the flexible 10-12-ish keyword count — both changes shift the score
distribution, and 0.35 may not be the right cutoff once they've landed.

## Retrieval flow
1. Generate `summary` + `keywords` + `confidence` for the new company via
   the summary call (summary is in-memory only, never persisted).
2. Embed the keyword list (e.g. `", ".join(keywords)`), query Chroma for
   top-k nearest existing companies by cosine similarity.
3. Filter to matches >= threshold. If NONE clear it, pass zero few-shot
   examples to the tagging prompt — do not force-feed weak/unrelated
   matches, and do not lower the bar just to have "something" to show.
4. Pass the surviving matches (name, their stored keywords, primary_tag,
   secondary_tags — NOT a summary, since none is stored for past
   companies) as few-shot context to build_tagging_prompt.

## Validation before trusting a change here
Test against a known set with obvious expected relationships: Amazon /
eBay / Flipkart should mutually score as similar (all e-commerce
marketplaces); Razorpay / Stripe should mutually score as similar (both
payment infrastructure); cross-pairs (Amazon vs Razorpay) should score
lower than same-industry pairs. If a cross-pair outranks a same-industry
pair, something in embedding input or distance space is wrong — do not
just retune the threshold, find the actual cause first (this is exactly
how the L2-vs-cosine bug and the prose-vs-keyword bug were both found).