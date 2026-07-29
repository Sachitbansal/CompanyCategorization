"""
summary_prompt.py — prompt construction for the company-summary DeepSeek call.

Stage 1 of the pipeline. Takes whatever signals we managed to scrape (which
may be nothing at all) plus the brand name, and returns THREE fields in one
call: `summary`, `keywords`, `confidence`.

What each field is actually for — these have very different lifespans:
  - `summary`: 50-100 word plain-prose description. EPHEMERAL. It is used
    in-memory only, twice, within this single request: as the model's own
    working context for producing `keywords` in this same call, and as the
    "COMPANY TO CATEGORIZE" block of the tagging call immediately after.
    It is never written to SQLite, never embedded, never retrievable later.
  - `keywords`: industry/sector/product terms. This is the ONLY field that
    gets embedded (all-MiniLM-L6-v2 -> Chroma) and the only thing that
    drives top-k similar-company retrieval. It is also what past companies
    carry as their stored representation in the few-shot block.
  - `confidence`: "high" | "low", binary. main.py branches on "low" to ask
    the user for more context and loop back into this call once.

Why keywords and not prose get embedded: prose summaries were matching on
shared sentence boilerplate ("Indian", "serves businesses of all sizes",
"competes in the X space") rather than on the actual business. Confirmed
live — Razorpay outranked Flipkart for a Flipkart query, and Amazon/Flipkart
(same industry) scored below Amazon/Razorpay (different industries). See
.claude/skills/tag-canonicalization/SKILL.md.

Two things this file deliberately does NOT do:
  - It does not call the API. temperature=0 and response_format={"type":
    "json_object"} are the caller's (main.py's) job.
  - It does not decide what to do with low confidence. Per CLAUDE.md, a
    low-confidence result on a failed scrape means "ask the user for more
    context and loop back into the summary call once" — that control flow
    lives in main.py.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid importing requests/bs4 just to build a string
    from scraper import ScrapedContent


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You are a business analyst who profiles companies for a brand-categorization system.

Your job is to work out what a company actually does — its industry, its products or services, who its customers are (consumers, businesses, developers, etc.), and its business model where that is clear — and to report that as three fields: a short prose summary, a keyword list, and a confidence flag.

Rules for the summary field:
0. The summary is a scratchpad, not a deliverable. It is read once, inside this same request, by you (to produce the keyword list below) and by one downstream categorization step. It is then thrown away — never stored, never shown to a person, never retrieved again. So write it as a direct statement of fact about the company. Do not write for a future reader, do not refer to the summary itself ("as noted above", "this summary describes"), do not add framing, caveats, or a closing sentence.
1. Write 50-100 words as ONE continuous plain-prose paragraph. No bullet points, no headings, no line breaks, no markdown.
2. Lead with the concrete category of business. Start with the pattern "<Brand> is a ..." — for example "Zomato is an Indian online food delivery and restaurant discovery platform that ...".
3. Use neutral, descriptive, third-person language. Do not copy marketing slogans, taglines, or promotional adjectives ("world-class", "leading", "revolutionary", "seamless") from the source material. Rewrite them into plain factual statements or drop them.
4. Prefer concrete nouns over abstractions: name the products, the sector, the customer type. "sells running shoes and athletic apparel" is useful; "empowers people to move" is not.
5. Do not invent specifics you were not given and do not know: no fabricated founding years, headcounts, revenue figures, funding rounds, or executive names. Omitting a detail is always better than guessing one.
6. Do not mention the source of your information. Never write phrases like "according to the website", "the metadata says", "based on the title tag", or "I am not certain". Uncertainty is reported ONLY through the confidence field, never inside the summary text.
7. If the input signals are contradictory or the brand name is ambiguous (multiple real companies share it), describe the single most likely/most widely known interpretation and set confidence to "low".

Rules for the confidence field, which must be exactly "high" or "low":
- "high" — the signals given, and/or your own solid knowledge of this brand, clearly establish the industry and what the company sells. A downstream reader could categorize it correctly from your summary alone.
- "low" — you are guessing at the industry, the brand name is ambiguous or unknown to you, or the provided signals were too thin or too generic to identify the business (e.g. only a bare title, or copy that could describe any company).
Be honest here rather than agreeable. A "low" simply triggers a request for more context from the user; an over-confident "high" on a guessed summary silently corrupts the category assigned to this company and every company later matched against it.

Return ONLY a JSON object with exactly these two keys:
{"summary": "<50-100 word paragraph>", "confidence": "high" | "low"}"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_FIELD_LABELS = [
    ("title", "Page title"),
    ("meta_description", "Meta description"),
    ("og_description", "Open Graph description"),
    ("jsonld_description", "JSON-LD description"),
    ("h1", "Main heading (H1)"),
]


def _collect_signals(scraped: Optional["ScrapedContent"]) -> list[tuple[str, str]]:
    """
    Flatten a ScrapedContent into a list of (label, value) pairs, skipping
    anything missing or blank.

    Handles all three "we got nothing" cases identically: scraped is None,
    scraped.success is False, or success is True but every extracted field
    came back None (a page that parsed fine but carries no descriptive
    metadata). All three produce an empty list, which is what flips the
    prompt into name-only mode.
    """
    if scraped is None or not getattr(scraped, "success", False):
        return []

    signals: list[tuple[str, str]] = []
    for attr, label in _FIELD_LABELS:
        value = getattr(scraped, attr, None)
        if isinstance(value, str) and value.strip():
            signals.append((label, " ".join(value.split())))
    return signals


def build_summary_prompt(
    brand_name: str,
    scraped: Optional["ScrapedContent"],
    extra_context: str | None = None,
) -> str:
    """
    Build the user-message text for the summary call.

    Produces one of two prompt shapes:

    1. Signals available — lists each populated scraped field under a
       labelled heading and asks the model to combine those signals with
       its own knowledge of the brand. Only populated fields appear; a site
       with just a <title> and an <h1> yields a two-line signal block, and
       the model is explicitly told the block is partial so it does not
       treat missing metadata as meaningful.

    2. No signals (scrape failed, or returned nothing usable) — states
       plainly that this is a name-only summary and instructs the model to
       work from its own knowledge, or to set confidence "low" if the brand
       is unfamiliar or ambiguous. This is the branch that feeds CLAUDE.md's
       "ask the user for more context, loop back once" path.

    `extra_context` is that second pass: free-text the user supplied after a
    low-confidence name-only result. It is inserted as its own clearly
    labelled, highest-authority block so it outranks a thin scrape rather
    than competing with it.
    """
    signals = _collect_signals(scraped)
    parts: list[str] = [f"Brand name: {brand_name}"]

    if signals:
        signal_lines = "\n".join(f"- {label}: {value}" for label, value in signals)
        parts.append(
            "Signals extracted from the company's own website (these are the "
            "only fields that were present; missing fields simply were not on "
            "the page and mean nothing about the business):\n"
            f"{signal_lines}"
        )
        parts.append(
            "Write the summary primarily from these signals. Website copy is "
            "often promotional and vague — where it is, lean on your own "
            "knowledge of this brand to state plainly what the company sells "
            "and to whom. If the signals are all generic marketing language "
            "and you do not independently recognize the brand, set confidence "
            'to "low".'
        )
    else:
        parts.append(
            "No website content could be retrieved for this brand (the site "
            "was unreachable, blocked, timed out, or contained no descriptive "
            "metadata). You are working from the brand name alone."
        )
        parts.append(
            "Write the summary from your own knowledge of this brand name. If "
            "you recognize it confidently, describe it normally and set "
            'confidence to "high". If you do not recognize it, or several '
            "different companies plausibly share this name, describe the most "
            'likely interpretation and set confidence to "low" — do not '
            "invent a plausible-sounding company."
        )

    if extra_context:
        parts.append(
            "Additional context supplied by the user about this brand. Treat "
            "this as the most reliable input available and let it override any "
            "conflicting assumption:\n"
            f"{' '.join(extra_context.split())}"
        )

    parts.append(
        "Return the JSON object now, with the 50-100 word summary paragraph "
        "and the confidence value."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

SUMMARY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "50-100 word plain-prose paragraph describing what the "
                "company does. No markdown, no line breaks, no hedging "
                "language about sources or certainty."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "low"],
            "description": (
                "'high' if the industry and offering are clearly established "
                "by the signals or by solid knowledge of the brand; 'low' if "
                "the summary is a guess, the brand is ambiguous, or the "
                "signals were too thin to identify the business."
            ),
        },
    },
    "required": ["summary", "confidence"],
    "additionalProperties": False,
}
