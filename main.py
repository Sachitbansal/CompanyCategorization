"""
main.py — orchestration for the optimiseGEO brand categorization pipeline.

run_pipeline(name_or_url) is the core entry point: a generator that yields
one event dict per pipeline stage so a caller (CLI, Streamlit, anything
else) can show live progress. It wires together every other module in the
order locked by CLAUDE.md:

    1. Exact-match fast path (website, then name) -> zero LLM/embedding
       cost on repeat lookups.
    2. Scrape the website (scraper.py); on failure, summarize from the
       brand name alone.
    3. DeepSeek summary call, temperature=0, structured JSON
       (prompts/summary_prompt.py), returning THREE fields: `summary`
       (ephemeral — used only in-memory for this one request, as context
       for keyword generation and the tagging call right after, then
       discarded; never stored, never embedded), `keywords` (the only
       thing that IS persisted and embedded), and `confidence`. Whenever
       confidence comes back low — whether that's because the scrape
       failed and the model doesn't know the brand by name alone, or the
       scrape succeeded but the page content still wasn't enough to pin
       down what the company does — the generator PAUSES (yields
       "awaiting_context" and stops) so the caller can ask the user to
       describe the brand themselves. Calling run_pipeline again with
       that description as `extra_context` resumes and retries the
       summary call ONCE (not indefinitely) with it folded in.
    4. Embed `keywords` and retrieve top-k similar companies as few-shot
       context (embed.py) — empty list if nothing clears the threshold.
    5. DeepSeek tag-assignment call (prompts/tagging_prompt.py).
    6. Run every proposed tag through the string-level dedup guardrail
       (dedup.py) before touching the categories table — this is the
       only source of truth for "is this tag new", never the LLM.
    7. Write the company row, junction rows, and embedding to storage.

categorize_brand(name_or_url) is a blocking convenience wrapper around
run_pipeline for terminal use — it drives the generator and falls back to
input() when the pipeline pauses for more context.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import db
import dedup
import embed
import scraper
from prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, build_summary_prompt
from prompts.tagging_prompt import TAGGING_SYSTEM_PROMPT, build_tagging_prompt

load_dotenv()

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Human-readable labels for each pipeline step, keyed by the "step" value
# in the events run_pipeline yields. Shared by any UI that wants to show
# progress without hardcoding step names twice.
STEP_LABELS = {
    "cache_check": "Checking for an existing record",
    "scrape": "Scraping website",
    "summarize": "Generating company summary",
    "retrieve_similar": "Retrieving similar companies",
    "assign_tags": "Assigning tags",
    "dedup_tags": "Running dedup guardrail",
    "save": "Saving to database",
}


def _client() -> OpenAI:
    """DeepSeek exposes an OpenAI-compatible API, so the OpenAI SDK just
    needs its base_url pointed elsewhere and the DeepSeek key in its place."""
    return OpenAI(api_key=os.environ["KEY"], base_url=DEEPSEEK_BASE_URL)


def _call_json(client: OpenAI, system_prompt: str, user_prompt: str) -> dict:
    """
    One structured DeepSeek call: temperature=0 for determinism (the
    consistency requirement starts here, before dedup.py even runs) and
    response_format=json_object so the reply is guaranteed parseable JSON
    matching the shape described in the system prompt.
    """
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def _looks_like_url(name_or_url: str) -> bool:
    """
    Auto-detects whether the input is a website or a bare brand name:
    anything with a dot and no spaces ("stripe.com", "www.acme.io") is
    treated as a URL; anything else ("Stripe", "Acme Corp") is treated as
    a name-only lookup. No explicit "is this a URL?" flag needed from the
    caller.
    """
    return "." in name_or_url and " " not in name_or_url


def _brand_key(s: str) -> str:
    """
    Reduce a name or website to a bare core token for fast-path matching:
    "https://www.amazon.com/" -> "amazon", "Amazon" -> "amazon". Without
    this, a company first looked up as "amazon.com" and later as "amazon"
    fail both find_company_by_website (no dot -> nothing to compare) and
    find_company_by_name (name column literally holds "amazon.com", which
    != "amazon") and get inserted as a duplicate row. Taking just the
    first label before any dot collapses both spellings, and any other
    TLD/subdomain variant, onto the same key.
    """
    s = s.strip().lower()
    s = s.removeprefix("https://").removeprefix("http://").removeprefix("www.")
    s = s.rstrip("/")
    return s.split(".")[0] if s else s


def _find_existing_company(conn, name_or_url: str) -> dict | None:
    """
    Fast-path lookup by core brand key across every existing company's
    name AND website, not just an exact match on one column. Still pure
    SQLite reads, zero LLM/embedding cost — just a slightly fuzzier key
    than a raw exact match, specifically to catch "amazon" vs
    "amazon.com" vs "www.amazon.com" all being the same company.
    """
    key = _brand_key(name_or_url)
    if not key:
        return None
    for company in db.get_all_companies(conn):
        if _brand_key(company["name"]) == key:
            return company
        if company["website"] and _brand_key(company["website"]) == key:
            return company
    return None


def _resolve_tag(conn, proposed: str, categories: list[dict]) -> tuple[int, str, list[dict]]:
    """
    Run one proposed tag through the dedup guardrail and return
    (category_id, canonical_name, updated_categories).

    `categories` is threaded through and appended to in-memory (rather
    than re-querying the DB) so that if this same tagging call proposes
    two new tags that are near-duplicates of each other (e.g. primary
    "Fintech" and a secondary that's a reworded "Fintech"), the second
    one is deduped against the first even though neither is in the DB yet.
    """
    match = dedup.find_canonical_match(proposed, categories)
    if match is not None:
        return match["id"], match["name"], categories

    new_id = db.insert_category(conn, proposed)
    new_category = {"id": new_id, "name": proposed, "primary_count": 0, "secondary_count": 0}
    return new_id, proposed, categories + [new_category]


def _company_result(conn, company: dict, cache_hit: bool) -> dict:
    primary = db.find_category_by_id(conn, company["primary_tag_id"])
    secondaries = db.get_secondary_tags(conn, company["id"])
    return {
        "company_id": company["id"],
        "name": company["name"],
        "website": company["website"],
        "keywords": company["keywords"],
        "primary_tag": primary["name"] if primary else None,
        "secondary_tags": [s["name"] for s in secondaries],
        "cache_hit": cache_hit,
    }


def run_pipeline(name_or_url: str, extra_context: str | None = None, scraped=None):
    """
    Generator driving the full pipeline. Yields one event dict per stage:
        {"step": <str>, "status": "running"|"done", "detail": <str, optional>}
    plus two terminal events:
        {"step": "awaiting_context", "status": "waiting", "brand_name": ...,
         "draft_summary": ..., "scraped": <ScrapedContent or None>}
            -> the generator stops here without saving anything. Call
               run_pipeline again with the SAME name_or_url, the
               `scraped` object handed back here (so it isn't re-fetched),
               and `extra_context` set to either the user's description or
               "" (empty string, meaning "asked, user declined, proceed
               anyway with the original summary") to resume.
        {"step": "complete", "status": "done", "result": {...}}
            -> the final dict (same shape returned by categorize_brand).

    `extra_context=None` means "haven't asked the user yet" and is what
    allows the low-confidence pause to trigger; `extra_context=""` means
    "already asked, proceed regardless of confidence" — this distinction
    is what stops the pipeline from asking twice.
    """
    conn = db.init_db()

    yield {"step": "cache_check", "status": "running"}
    existing = _find_existing_company(conn, name_or_url)
    if existing is not None:
        yield {"step": "cache_check", "status": "done", "detail": "existing record found"}
        yield {"step": "complete", "status": "done", "result": _company_result(conn, existing, cache_hit=True)}
        return
    yield {"step": "cache_check", "status": "done", "detail": "no existing record"}

    client = _client()
    brand_name = name_or_url
    website = name_or_url if _looks_like_url(name_or_url) else ""

    yield {"step": "scrape", "status": "running", "detail": website or "no website given, using name only"}
    if scraped is None:
        scraped = scraper.scrape_website(website) if website else None
    scrape_ok = scraped is not None and getattr(scraped, "success", False)
    yield {"step": "scrape", "status": "done", "detail": "success" if scrape_ok else "no usable content"}

    yield {"step": "summarize", "status": "running"}
    prompt = build_summary_prompt(brand_name, scraped, extra_context=extra_context or None)
    summary_data = _call_json(client, SUMMARY_SYSTEM_PROMPT, prompt)
    yield {"step": "summarize", "status": "done", "detail": f"confidence={summary_data.get('confidence')}"}

    if summary_data.get("confidence") == "low" and extra_context is None:
        yield {
            "step": "awaiting_context",
            "status": "waiting",
            "brand_name": brand_name,
            "draft_summary": summary_data["summary"],
            "scraped": scraped,
        }
        return

    # `summary` stays in-memory for this request only (context for the
    # tagging call's "COMPANY TO CATEGORIZE" block right below) — it is
    # never stored or embedded. `keywords` is what gets persisted and
    # embedded; it's the only thing that exists for past companies too.
    summary = summary_data["summary"]
    keywords = summary_data["keywords"]

    yield {"step": "retrieve_similar", "status": "running"}
    embedder = embed.get_embedder()
    collection = embed.get_chroma_collection()
    similar = embed.top_k_similar(collection, embedder, keywords)

    few_shot_examples = []
    for match in similar:
        company = db.get_company_by_id(conn, match["company_id"])
        if not company:
            continue
        primary = db.find_category_by_id(conn, company["primary_tag_id"])
        secondaries = db.get_secondary_tags(conn, company["id"])
        few_shot_examples.append(
            {
                "name": company["name"],
                "keywords": company["keywords"],
                "primary_tag": primary["name"] if primary else "",
                "secondary_tags": [s["name"] for s in secondaries],
                "similarity": match["similarity"],
            }
        )
    yield {
        "step": "retrieve_similar",
        "status": "done",
        "detail": f"{len(few_shot_examples)} similar companies",
        "matches": few_shot_examples,
    }

    yield {"step": "assign_tags", "status": "running"}
    categories = db.get_all_categories(conn)
    category_names = [c["name"] for c in categories]
    tagging_prompt = build_tagging_prompt(summary, keywords, category_names, few_shot_examples)
    tag_data = _call_json(client, TAGGING_SYSTEM_PROMPT, tagging_prompt)
    yield {"step": "assign_tags", "status": "done", "detail": f"proposed primary={tag_data['primary_tag']!r}"}

    yield {"step": "dedup_tags", "status": "running"}
    primary_id, primary_name, categories = _resolve_tag(conn, tag_data["primary_tag"], categories)
    db.increment_tag_count(conn, primary_id, "primary")

    secondary_ids: list[int] = []
    secondary_names: list[str] = []
    for proposed in tag_data.get("secondary_tags", [])[:3]:
        sec_id, sec_name, categories = _resolve_tag(conn, proposed, categories)
        if sec_id == primary_id or sec_id in secondary_ids:
            continue  # guard against the model repeating/near-duplicating the primary
        secondary_ids.append(sec_id)
        secondary_names.append(sec_name)
        db.increment_tag_count(conn, sec_id, "secondary")
    yield {
        "step": "dedup_tags",
        "status": "done",
        "detail": f"primary={primary_name!r}, secondary={secondary_names}",
    }

    yield {"step": "save", "status": "running"}
    company_id = db.insert_company(conn, brand_name, website, keywords, primary_id, embedding_id="")
    embedding_id = embed.embed_and_store(collection, embedder, company_id, keywords)
    db.update_company_embedding_id(conn, company_id, embedding_id)
    if secondary_ids:
        db.add_secondary_tags(conn, company_id, secondary_ids)
    yield {"step": "save", "status": "done", "detail": f"company_id={company_id}"}

    result = {
        "company_id": company_id,
        "name": brand_name,
        "website": db.normalize_website(website) if website else "",
        "keywords": keywords,
        "primary_tag": primary_name,
        "secondary_tags": secondary_names,
        "cache_hit": False,
    }
    yield {"step": "complete", "status": "done", "result": result}


def categorize_brand(name_or_url: str) -> dict:
    """
    Blocking terminal entry point. Drives run_pipeline to completion,
    using input() when it pauses to ask for more context.
    """
    extra_context = None
    scraped = None

    while True:
        result = None
        pending = None

        for event in run_pipeline(name_or_url, extra_context=extra_context, scraped=scraped):
            step = event["step"]
            if step == "awaiting_context":
                pending = event
                break
            if step == "complete":
                result = event["result"]
            elif event["status"] in ("running", "done"):
                label = STEP_LABELS.get(step, step)
                marker = "..." if event["status"] == "running" else f" ({event.get('detail', 'done')})"
                print(f"[{step}] {label}{marker}")

        if result is not None:
            return result

        # pending: pipeline paused for a low-confidence, name-only-or-thin summary
        scraped = pending["scraped"]
        extra_context = input(
            f"\nLow-confidence summary for '{pending['brand_name']}':\n"
            f"  \"{pending['draft_summary']}\"\n"
            "Describe what this brand does (industry, product, customers), "
            "or press Enter to keep this summary as-is: "
        ).strip()
        # "" (falsy but not None) tells the next pass "already asked, don't ask again"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <brand name or website>")
        sys.exit(1)

    final_result = categorize_brand(" ".join(sys.argv[1:]))
    print(json.dumps(final_result, indent=2))
