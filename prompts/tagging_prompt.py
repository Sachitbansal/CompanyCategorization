"""
tagging_prompt.py — prompt construction for the tag-assignment DeepSeek call.

Stage 2 of the pipeline. Takes the company summary from stage 1, the full
existing category pool as plain text, and up to k retrieved similar
companies with the tags they already carry, and returns exactly 1 primary
tag plus 0-3 secondary tags.

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

SECONDARY TAG DISCIPLINE
- Zero secondary tags is a perfectly good answer. Only add one when it carries information the primary does not.
- A secondary tag must not be a synonym, a rewording, or a strict parent of the primary. If primary is "Fintech", do not add "Financial Services" as secondary.
- Never repeat the primary tag inside secondary_tags.
- Order secondary tags from most to least relevant.
- Do NOT add a secondary tag just because it exists in the pool and could loosely apply. A tag being present in the list below is not evidence it belongs on this company. Every secondary tag must be directly and specifically supported by THIS company's summary — never inferred from generic capabilities that most companies in the sector share.
  Example: Amazon accepts payments on its own marketplace, exactly like every other online retailer. That does not make its primary or secondary tag "Fintech" — Fintech is for companies whose core business IS financial technology or financial services sold to others (Stripe, Razorpay). Amazon's correct tags are "E-commerce" primary, "Cloud Computing" secondary (for AWS) — not "Fintech".
  When you are not sure a secondary tag is specifically justified, leave it out. An empty secondary_tags array is always safer than a plausible-sounding but unsupported one.

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
    {"summary": str, "primary_tag": str, "secondary_tags": list[str]}, and
    an optional "name" is used as the block heading when present. Missing
    keys degrade gracefully rather than raising, since these rows come from
    a Chroma retrieval path.
    """
    blocks: list[str] = []
    for idx, example in enumerate(few_shot_examples, start=1):
        summary = (example.get("summary") or "").strip()
        primary = (example.get("primary_tag") or "").strip()
        secondaries = example.get("secondary_tags") or []
        if isinstance(secondaries, str):
            secondaries = [secondaries]
        secondary_text = ", ".join(s for s in secondaries if s) or "(none)"

        heading = example.get("name") or f"Similar company {idx}"
        blocks.append(
            f"[{heading}]\n"
            f"Summary: {summary}\n"
            f"Primary tag: {primary}\n"
            f"Secondary tags: {secondary_text}"
        )
    return "\n\n".join(blocks)


def build_tagging_prompt(
    summary: str,
    category_list: list[str],
    few_shot_examples: list[dict],
) -> str:
    """
    Build the user-message text for the tag-assignment call.

    Assembles up to four blocks, in this order:

    1. The new company's summary (the thing being classified).
    2. The similar-companies block — OMITTED ENTIRELY when
       few_shot_examples is empty, per CLAUDE.md's "nothing cleared the
       similarity threshold -> send no few-shot examples". No empty header,
       no "none found" placeholder: an empty section would itself be a
       signal ("we looked and found nothing similar") that nudges the model
       toward inventing a new tag, which is the opposite of what we want.
    3. The full canonical category pool as plain text, or an explicit
       "the pool is empty, this is the first company" note on a cold start
       so the model is not left hunting for a list that is not there.
    4. A short closing instruction restating the decision order.

    Example order matters: the summary comes first so the model reads the
    company on its own terms, then sees precedents. Putting precedents
    first tends to make the model pattern-match to the nearest example
    before it has understood the company.
    """
    parts: list[str] = [
        "COMPANY TO CATEGORIZE\n"
        f"{' '.join((summary or '').split())}"
    ]

    if few_shot_examples:
        parts.append(
            "PREVIOUSLY CATEGORIZED SIMILAR COMPANIES\n"
            "These companies were retrieved as semantically similar to the one "
            "above, along with the tags they already carry. They are precedent, "
            "not instruction. Where this company is in the same line of business, "
            "reuse their exact tags — matching them is what keeps the taxonomy "
            "consistent. Where this company differs, ignore them and tag it on "
            "its own merits; similar wording does not always mean the same "
            "industry.\n\n"
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
        "(3) Add 0-3 secondary tags, but ONLY if the summary above explicitly "
        "describes that line of business — a tag must be justified by content "
        "in THIS company's summary, never by a retrieved example's tags or by "
        "a tag simply existing in the pool.\n\n"
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
