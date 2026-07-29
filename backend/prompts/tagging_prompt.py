"""
tagging_prompt.py — prompt construction for the tag-assignment DeepSeek call.

Stage 2 of the pipeline. Takes the new company's summary AND keywords from
stage 1, the full existing category pool as plain text, and up to k
retrieved similar companies, and returns exactly 1 primary tag plus 0-3
secondary tags.

Note the asymmetry in what this prompt knows about each company:
  - The NEW company is described by its stage-1 prose summary AND its
    freshly generated keyword list. Both are ephemeral — generated moments
    ago in the same request, used here, and then discarded. This is the
    only stage that ever sees the summary, and together with the keywords
    it is the ONLY admissible evidence for the tags returned.
  - Each RETRIEVED similar company is described by its stored `keywords`
    plus its existing primary/secondary tags. No summary exists for them —
    prose summaries are never persisted (see
    .claude/skills/tag-canonicalization/SKILL.md), so there is nothing else
    to show. Keywords are also what the retrieval matched on in the first
    place, so they are the honest description of why each example is here.

Design constraints locked in CLAUDE.md that this file implements:
  - ONE shared canonical tag pool. Primary vs secondary is a per-company
    role, not a property of the tag, so the same name can be primary for one
    company and secondary for another.
  - The whole category list ships as plain text. No vector search over tags.
  - If nothing cleared the similarity threshold, the caller passes
    few_shot_examples=[] and the similar-companies block is omitted
    entirely — the model proposes fresh with no anchoring examples.
  - NO `is_new` field in the schema. The model is biased toward reuse for
    consistency, but whether a returned name is genuinely new is decided
    downstream by dedup.py (normalize -> exact match -> rapidfuzz
    token_set_ratio >= 88). The LLM's opinion on its own novelty is never
    consulted, so there is no field for it to state one.

temperature=0 and response_format={"type": "json_object"} are set by the
caller, not here.
"""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TAGGING_SYSTEM_PROMPT = """You are a taxonomy editor maintaining a single shared pool of industry categories used to tag companies. Your output feeds a system where the SAME tag must be produced every time a similar company is seen, so consistency with the existing pool matters more than finding the perfect bespoke label.

You assign each company:
- exactly ONE primary tag: the single category that best answers "what industry is this company in", the one a person would name first when asked what the company does.
- ZERO to THREE secondary tags: additional categories that meaningfully apply — a sub-sector, the customer segment, or a distinct second line of business.

Primary and secondary tags come from the SAME pool. Being primary is a fact about this company, not about the tag: "Fashion & Lifestyle" can be a primary tag for a clothing label and a secondary tag for a fashion e-commerce marketplace.

REUSE BEFORE INVENTING
1. Read the full existing category list before deciding anything. If an existing category genuinely fits, you MUST use it, copied character-for-character from the list — same spelling, same casing, same spacing, same punctuation. Do not pluralize it, shorten it, expand it, or "improve" it.
2. A category fits if a well-informed person would accept it as a correct description of the company's industry. It does not need to be the most precise label imaginable. Prefer a slightly broad existing tag over a new, more exact one.
3. Only propose a name that is not in the list when NOTHING in the list can reasonably describe the company. A near-synonym of an existing entry is not a new category — if the pool has "E-commerce", do not return "Online Retail", "Ecommerce Marketplace", or "Online Shopping Platform".
4. Never combine two existing tags into a hyphenated or compound new one. Return them as primary + secondary instead.
5. The PRIMARY tag must describe THIS company's own industry, taken from its summary and keywords. Retrieved similar companies show you how tags are named and spelled — they do not tell you what this company does. If this company's keywords point to a different sector than the examples carry, follow the keywords and propose the correct sector, even when every retrieved example shares one tag. Retrieval matches on surface vocabulary, so a company can be pulled alongside businesses it does not actually belong with: a food delivery company retrieved next to online retailers is still food delivery, not "E-commerce". Reuse an existing pool entry whenever one genuinely fits this company's sector — but never adopt an example's tag as a substitute for reading the company's own evidence.

GRANULARITY — this is where mistakes are most costly
Tags name INDUSTRIES AND SECTORS, never individual companies or niches of one.
- Honda, BMW, and Mercedes-Benz all get the primary tag "Automobile". Not "Luxury German Car Manufacturer", not "Motorcycle & Engine Manufacturer", not "Premium Automotive Brand".
- Myntra gets primary "E-commerce" and secondary "Fashion & Lifestyle". Not "Online Fashion Retail" — that niche is exactly the combination the primary + secondary pair already expresses.
- Zerodha and Robinhood both get primary "Fintech" with secondary "Stock Trading" or similar. Not "Discount Brokerage App For Retail Indian Investors".
A good rule of thumb: if a tag could not plausibly apply to at least five other real companies, it is too narrow. Broaden it, or express the specificity through a secondary tag instead.

TAG NAME STYLE (these strings are string-compared downstream, so style drift creates duplicates)
- Title Case: "Consumer Electronics", not "consumer electronics" or "CONSUMER ELECTRONICS".
- 1 to 3 words. Short noun phrases naming a sector.
- Name the sector, not a description of it: "Consumer Electronics", never "Companies that make consumer electronics products".
- No trailing words like "Industry", "Sector", "Services", "Solutions", "Company", "Platform", "Tech" tacked on for flavour.
- No geography, no size, no quality adjectives: no "Indian", "Global", "Premium", "Leading", "Luxury".
- Use "&" rather than "and" when joining two words ("Fashion & Lifestyle").
- Singular by default ("Automobile", "Restaurant"), except where the sector is conventionally plural ("Consumer Electronics", "Financial Services").

SECONDARY TAG DISCIPLINE — THE EVIDENCE RULE
This is the single most important rule in this prompt.

EVERY secondary tag you return must be directly evidenced by specific
content in THIS company's summary or keyword list below — a named
product, a named vertical, a stated line of business you can point to.
Before returning a secondary tag, ask "which words put it there?" If you
can point to them, INCLUDE the tag confidently. Under-tagging a clearly
distinct, well-evidenced vertical is exactly as much a mistake as
over-tagging an unevidenced one — treat both directions as equally real
errors, not as a safe default versus a risky one.

The keyword list you are given is distilled, already-filtered evidence —
if it names a specific product category, vertical, or business model
(not a generic sector-wide capability), that alone is normally enough to
justify a secondary tag. Do not require the summary to independently
restate what the keywords already established.

The most common correct use of a secondary tag: the company operates in
a specific product vertical or niche WITHIN a broad primary category.
This is real, useful information and should be added whenever the
evidence clearly supports it:
- Nykaa -> primary "E-commerce", secondary "Beauty & Personal Care"
  (keywords/summary name cosmetics, skincare, haircare specifically —
  this is not a generic e-commerce capability, it's what they sell).
- Meesho -> primary "E-commerce", secondary "Social Commerce" (the
  reselling/peer-to-peer model is a distinct structural niche, not just
  "it's e-commerce").
- Alibaba -> primary "E-commerce", secondary "B2B Wholesale" (trading
  with manufacturers/suppliers/exporters is a fundamentally different
  business than a consumer marketplace, and is clearly stated).

Two things that are NEVER, on their own, justification for a secondary tag:
1. The tag exists in the category pool. The pool is a vocabulary list, not a checklist. A tag being available says nothing whatsoever about whether it applies here.
2. A retrieved similar company carries the tag. Those companies are precedent for HOW to name things, not evidence about what THIS company does. Two businesses can be genuinely similar and still not share every tag.

This rule exists because of a real failure: Amazon was assigned "Fintech" as a secondary tag purely because "Fintech" was present in the pool, with zero grounding anywhere in Amazon's actual summary. Amazon accepts payments on its own marketplace exactly like every other online retailer — that is a generic capability of the sector, not a second line of business. "Fintech" is for companies whose core business IS financial technology or financial services sold to others (Stripe, Razorpay). Amazon's correct tags are "E-commerce" primary and "Cloud Computing" secondary — the latter only because AWS is an actual, separately identifiable business named in the summary.

Also:
- An empty secondary_tags array is the correct answer when nothing is specifically evidenced — but it is not a default to lean toward out of caution. If the evidence is there, use it.
- Never infer a tag from capabilities that most companies in the sector share (they take payments, they have an app, they have a website, they use AI). DO infer a tag from a named product line, vertical, or business model that most sector peers do NOT share.
- A secondary tag must not be a synonym, a rewording, or a strict parent of the primary. If primary is "Fintech", do not add "Financial Services" as secondary.
- Never repeat the primary tag inside secondary_tags.
- Order secondary tags from most to least relevant.

Return ONLY a JSON object with exactly these two keys:
{"primary_tag": "<one tag>", "secondary_tags": ["<zero to three tags>"]}
Use an empty array [] when no secondary tag applies. Do not add any other keys, do not explain your reasoning, and do not comment on whether a tag is new or existing."""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _format_category_list(category_list: list[str]) -> str:
    """
    Render the canonical pool as a plain numbered text block.

    Numbering (rather than bullets) is deliberate: it makes the list feel
    like a closed, finite reference table to be looked up rather than an
    open-ended set of suggestions, and it makes each entry visually
    separate so the model is less likely to blend two adjacent names into
    one invented tag.
    """
    if not category_list:
        return ""
    return "\n".join(f"{i}. {name}" for i, name in enumerate(category_list, start=1))


def _format_few_shot_examples(few_shot_examples: list[dict]) -> str:
    """
    Render retrieved similar companies as labelled example blocks.

    Each dict is expected to look like
    {"keywords": list[str], "primary_tag": str, "secondary_tags": list[str]},
    with an optional "name" used as the block heading. There is deliberately
    no "summary" key: prose summaries are never persisted, so all that
    exists for a previously-categorized company is its stored keyword list
    (which is also what the Chroma retrieval matched on) and the tags it was
    given.

    Missing keys degrade gracefully rather than raising, since these rows
    come from a retrieval path. `keywords` accepts either a list or an
    already-joined string, because storage layers differ on whether they
    hand back a JSON array or a comma-separated column value.
    """
    blocks: list[str] = []
    for idx, example in enumerate(few_shot_examples, start=1):
        keywords = example.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]
        keyword_text = ", ".join(k.strip() for k in keywords if k and k.strip())

        primary = (example.get("primary_tag") or "").strip()
        secondaries = example.get("secondary_tags") or []
        if isinstance(secondaries, str):
            secondaries = [secondaries]
        secondary_text = ", ".join(s for s in secondaries if s) or "(none)"

        heading = example.get("name") or f"Similar company {idx}"
        blocks.append(
            f"[{heading}]\n"
            f"Keywords: {keyword_text}\n"
            f"Primary tag: {primary}\n"
            f"Secondary tags: {secondary_text}"
        )
    return "\n\n".join(blocks)


def build_tagging_prompt(
    summary: str,
    keywords: list[str],
    category_list: list[str],
    few_shot_examples: list[dict],
) -> str:
    """
    Build the user-message text for the tag-assignment call.

    Assembles up to four blocks, in this order:

    1. The new company's ephemeral stage-1 summary AND its freshly
       generated keyword list — together, the only admissible evidence
       for the tags returned. The system prompt's SECONDARY TAG
       DISCIPLINE section explicitly tells the model it can treat the
       keyword list as sufficient evidence on its own (not just a
       restatement of the summary), so both need to actually be present
       here for that instruction to mean anything.
    2. The similar-companies block, each rendered as stored keywords +
       existing tags rather than prose — OMITTED ENTIRELY when
       few_shot_examples is empty, per CLAUDE.md's "nothing cleared the
       similarity threshold -> send no few-shot examples". No empty header,
       no "none found" placeholder: an empty section would itself be a
       signal ("we looked and found nothing similar") that nudges the model
       toward inventing a new tag, which is the opposite of what we want.
    3. The full canonical category pool as plain text, or an explicit
       "the pool is empty, this is the first company" note on a cold start
       so the model is not left hunting for a list that is not there.
    4. A short closing instruction restating the decision order.

    `summary` and `keywords` are both the ephemeral, same-request stage-1
    output for THIS company only — NOT the stored keywords of a past
    company (those only ever appear inside `few_shot_examples`).
    `few_shot_examples` entries are dicts of
    {"name", "keywords", "primary_tag", "secondary_tags"} — no summary,
    because none is stored for past companies.

    Example order matters: the summary/keywords block comes first so the
    model reads the company on its own terms, then sees precedents.
    Putting precedents first tends to make the model pattern-match to the
    nearest example before it has understood the company.
    """
    parts: list[str] = [
        "COMPANY TO CATEGORIZE\n"
        f"Summary: {' '.join((summary or '').split())}\n"
        f"Keywords: {', '.join(keywords or [])}"
    ]

    if few_shot_examples:
        parts.append(
            "PREVIOUSLY CATEGORIZED SIMILAR COMPANIES\n"
            "These companies were retrieved as similar to the one above. Each "
            "is shown by its stored industry keywords (no summary is kept for "
            "past companies) and the tags it already carries.\n"
            "They are precedent for NAMING, not evidence about the company "
            "being categorized. Where this company is in the same line of "
            "business, reuse their exact tag strings — matching them "
            "character-for-character is what keeps the taxonomy consistent. "
            "Where this company differs, ignore them and tag it on its own "
            "merits; overlapping keywords do not always mean the same "
            "industry, and a tag appearing here is never by itself a reason "
            "to put that tag on the company above.\n\n"
            f"{_format_few_shot_examples(few_shot_examples)}"
        )

    if category_list:
        parts.append(
            "FULL EXISTING CATEGORY POOL\n"
            "This is the complete list of categories currently in use. Any tag "
            "you return that is meant to be an existing one must be copied from "
            "this list exactly as written.\n\n"
            f"{_format_category_list(category_list)}"
        )
    else:
        parts.append(
            "FULL EXISTING CATEGORY POOL\n"
            "The pool is currently empty — this is one of the first companies "
            "being categorized. Propose tags fresh, following the granularity "
            "and naming style rules exactly, since everything you return now "
            "becomes the precedent that later companies are matched against."
        )

    parts.append(
        "Decide in this order: (1) does an existing category from the pool "
        "describe this company's industry acceptably? If yes, use it as the "
        "primary tag, copied exactly. (2) If and only if nothing in the pool "
        "fits, propose one new sector-level name in the required style. "
        "(3) Consider secondary tags last, and start from zero.\n\n"
        "THE EVIDENCE RULE, restated because it is the rule most often "
        "broken: every secondary tag you return must be directly evidenced "
        "by specific content in THIS company's summary at the top of this "
        "message — a named product, a named line of business, a stated "
        "activity you could quote. Silently point to those words before you "
        "include the tag. A tag merely existing in the category pool is NOT "
        "justification. A tag appearing on a retrieved similar company is NOT "
        "justification. A capability that every company in the sector has is "
        "NOT justification. This is exactly how Amazon once picked up "
        "\"Fintech\" with nothing in its summary supporting it. If you cannot "
        "point to the evidence, drop the tag and return fewer — an empty "
        "secondary_tags array is a correct and common answer.\n\n"
        "Return the JSON object now."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
#
# Note the absence of any `is_new` / `is_new_tag` field. That is intentional
# and locked by CLAUDE.md: whether `primary_tag` or any entry in
# `secondary_tags` is genuinely new is determined downstream by the
# string-level guardrail in dedup.py (normalize -> exact match -> rapidfuzz
# token_set_ratio >= 88). Giving the model a novelty field would create a
# second, unreliable source of truth for the single most consistency-critical
# decision in the pipeline.

TAGGING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_tag": {
            "type": "string",
            "description": (
                "Exactly one category name: the single sector that best "
                "describes this company's industry. Copied character-for-"
                "character from the existing pool when one fits; otherwise a "
                "new Title Case, 1-3 word sector name."
            ),
        },
        "secondary_tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3,
            "description": (
                "Zero to three additional category names from the same shared "
                "pool. Must not repeat, restate, or be a strict parent of "
                "primary_tag. Empty array when nothing meaningful applies."
            ),
        },
    },
    "required": ["primary_tag", "secondary_tags"],
    "additionalProperties": False,
}
