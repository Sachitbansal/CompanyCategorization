"""
scraper.py — lightweight metadata scraper for Brand Categoriser.

Pulls only the handful of on-page signals that reliably describe "what a
company does": <title>, meta description, og:description, JSON-LD
description, and the first <h1>. Deliberately does NOT touch body text,
nav, or footer — that content is noisy and not needed to build a 50-100
word company summary downstream.

On any failure (bad URL, timeout, connection error, non-2xx status) this
returns a ScrapedContent with success=False and every field None. The
caller (not this file) is responsible for falling back to the LLM's own
knowledge of the brand name when that happens, per CLAUDE.md.
"""

from dataclasses import dataclass
import json

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; Brand Categoriser/1.0; +https://example.com)"
DEFAULT_TIMEOUT = 3.0


@dataclass
class ScrapedContent:
    title: str | None
    meta_description: str | None
    og_description: str | None
    jsonld_description: str | None
    h1: str | None
    success: bool


def _empty_failure() -> ScrapedContent:
    """Single place to construct the "nothing worked" result, so every
    failure path returns an identically-shaped object."""
    return ScrapedContent(
        title=None,
        meta_description=None,
        og_description=None,
        jsonld_description=None,
        h1=None,
        success=False,
    )


def extract_jsonld_description(soup: BeautifulSoup) -> str | None:
    """
    Look through every <script type="application/ld+json"> block on the
    page for a "description" field.

    JSON-LD is inconsistent in the wild, so we handle the common shapes
    without trying to be a full JSON-LD parser:
      - a single object: {"@type": "Organization", "description": "..."}
      - a list of objects: [{...}, {...}]
      - a wrapper with "@graph": {"@graph": [{...}, {...}]}

    We return the first non-empty "description" string we find, checking
    each candidate dict's top level and its "@graph" children (one level
    deep is enough — going further starts to be over-engineering for a
    field that's just a hint to fall back on).
    """
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        # Normalize to a flat list of candidate dicts to inspect.
        candidates = []
        if isinstance(data, dict):
            candidates.append(data)
            graph = data.get("@graph")
            if isinstance(graph, list):
                candidates.extend(item for item in graph if isinstance(item, dict))
        elif isinstance(data, list):
            candidates.extend(item for item in data if isinstance(item, dict))

        for item in candidates:
            description = item.get("description")
            if isinstance(description, str) and description.strip():
                return description.strip()

    return None


def scrape_website(url: str, timeout: float = DEFAULT_TIMEOUT) -> ScrapedContent:
    """
    Fetch `url` and pull out title/meta description/og:description/
    JSON-LD description/first h1.

    Design choices:
      - We prepend "https://" if the URL has no scheme, since brand
        websites are usually passed around as bare domains (e.g.
        "acme.com") rather than full URLs.
      - requests.get is wrapped in a broad try/except so timeouts,
        connection errors, redirect loops, DNS failures, etc. all funnel
        into the same success=False result instead of raising up to the
        caller — this keeps the pipeline resilient to any single bad
        site.
      - A non-200 status code (blocked, 404, 5xx, etc.) is treated the
        same as a network failure, since we can't trust the body of a
        non-success response to contain real content.
      - success=True means "we got a usable 200 response and parsed it",
        even if individual fields end up None because that particular
        site doesn't have, say, an og:description. It's the caller's job
        to decide whether an all-None-but-success result still counts as
        "not enough to summarize" and fall back accordingly.
    """
    if not url:
        return _empty_failure()

    if not url.lower().startswith(("http://", "https://")):
        url = f"https://{url}"

    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException:
        # Covers Timeout, ConnectionError, TooManyRedirects, SSLError,
        # and everything else requests can throw for a bad/unreachable URL.
        return _empty_failure()

    if response.status_code != 200:
        return _empty_failure()

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        # Malformed HTML shouldn't crash the pipeline either.
        return _empty_failure()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = None
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_description = meta_desc_tag["content"].strip() or None

    og_desc_tag = soup.find("meta", attrs={"property": "og:description"})
    og_description = None
    if og_desc_tag and og_desc_tag.get("content"):
        og_description = og_desc_tag["content"].strip() or None

    jsonld_description = extract_jsonld_description(soup)

    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(strip=True) if h1_tag else None

    return ScrapedContent(
        title=title,
        meta_description=meta_description,
        og_description=og_description,
        jsonld_description=jsonld_description,
        h1=h1,
        success=True,
    )
