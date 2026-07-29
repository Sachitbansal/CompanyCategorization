"""
dedup.py — string-level tag canonicalization guardrail for optimiseGEO.

Runs in plain Python after every DeepSeek tag-assignment call, before any
write to the categories table. This function is the SOLE source of truth
for "is this tag new" — there is no is_new field in the tagging JSON
schema (see prompts/tagging_prompt.py), and the LLM's own judgment about
novelty is never trusted, per CLAUDE.md.

Algorithm (per tag, primary or secondary):
    1. Normalize (lowercase, strip whitespace, strip hyphens, drop a
       small set of generic filler words like "platform"/"services").
    2. Exact match against every existing category's normalized name ->
       use the existing canonical string.
    3. No exact match -> rapidfuzz token_set_ratio >= 88 against the best
       candidate -> use the existing canonical string.
    4. Neither matches -> genuinely new category.
"""

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 88

# Generic suffix/filler words that add no distinguishing meaning to a tag
# ("Ecommerce Platform" and "Ecommerce" describe the same category; the
# LLM is instructed not to produce these per prompts/tagging_prompt.py,
# but this guardrail is the LAST line of defense, so it defends anyway).
_NOISE_WORDS = {
    "platform", "platforms",
    "solutions", "solution",
    "services", "service",
    "provider", "providers",
    "industry",
    "company", "companies",
}


def normalize_tag(tag: str) -> str:
    """
    Normalize a tag for comparison: lowercase, strip surrounding
    whitespace, remove hyphens ("E-Commerce" -> "ecommerce"), and drop
    generic filler words ("Ecommerce Platform" -> "ecommerce").

    If removing filler words would empty the token list (e.g. the tag IS
    just "Platform"), the original tokens are kept instead — a tag should
    never normalize to an empty string.
    """
    stripped = tag.strip().lower().replace("-", "")
    tokens = [t for t in stripped.split() if t]
    filtered = [t for t in tokens if t not in _NOISE_WORDS] or tokens
    return " ".join(filtered)


def find_canonical_match(
    proposed_tag: str,
    existing_categories: list[dict],
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> dict | None:
    """
    Decide whether `proposed_tag` is really one of `existing_categories`
    under different spelling/casing/hyphenation, or is genuinely new.

    Returns the matching category dict (caller should use its existing
    `name` verbatim, never insert a new row) or None (safe to insert
    `proposed_tag` as a brand-new category).

    `existing_categories` is the list of dicts returned by
    db.get_all_categories() — each needs at least a "name" key.

    Fuzzy step is gated on both categories normalizing to the SAME
    TOKEN COUNT before comparing scores. This matters because
    rapidfuzz.fuzz.token_set_ratio scores a tag whose words are a strict
    subset of another tag's words as ~100% similar regardless of what the
    extra word means — "SaaS" vs "B2B SaaS" scores ~100 on token_set_ratio
    the same way "Ecommerce" vs "Ecommerce Platform" does, but the first
    pair is two genuinely different categories (B2B is a substantive
    scope qualifier) while the second pair is one category with a filler
    suffix. Filler words are already stripped in normalize_tag, so by the
    time we're comparing token counts here, any remaining count mismatch
    means a real word was added or dropped — treated as "not the same
    tag" rather than risking a false collapse.
    """
    normalized_proposed = normalize_tag(proposed_tag)
    if not normalized_proposed:
        return None
    proposed_tokens = normalized_proposed.split()

    for category in existing_categories:
        if normalize_tag(category["name"]) == normalized_proposed:
            return category

    best_match = None
    best_score = -1

    for category in existing_categories:
        normalized_existing = normalize_tag(category["name"])
        existing_tokens = normalized_existing.split()

        if len(existing_tokens) != len(proposed_tokens):
            continue

        score = fuzz.token_set_ratio(normalized_proposed, normalized_existing)
        if score > best_score:
            best_score = score
            best_match = category

    if best_match is not None and best_score >= fuzzy_threshold:
        return best_match

    return None
