"""
db.py — SQLite persistence layer for Brand Categoriser.

Schema (see CLAUDE.md for the locked architecture):
    companies(id, name, website, keywords, primary_tag_id, embedding_id)
    categories(id, name, primary_count, secondary_count)
    company_tags(company_id, category_id)  -- junction table, secondary tags only

IMPORTANT: there is NO summary column. The prose summary used for keyword
generation and the tagging prompt is generated per-request and lives only
in the caller's memory — db.py never receives it, stores it, or returns
it. What IS persisted is `keywords`: the short keyword list derived from
that summary, shown on the Data page and embedded (embedding_id points at
the embedding of the keywords). Since SQLite has no native array type and
a full child table is overkill for a handful of short strings, `keywords`
is stored as JSON-encoded text (json.dumps(list[str])) and transparently
decoded back to a Python list by every read path in this module.

`website` is stored already-normalized (protocol/www/trailing slash
stripped, lowercased). Normalization can be done anywhere upstream, but
`normalize_website` is kept here too (existing callers such as main.py
import it from this module) and every write path in this module runs
values through it defensively so storage stays consistent even if a
caller forgets.

Indexes:
    companies(website)      -> exact-match fast path on website
    companies(LOWER(name))  -> exact-match fast path on name (fallback)

Exact-match lookup:
    exact_match_lookup(conn, website, name) is the ONE function callers
    should use for the fast-path dedup check. It checks BOTH the website
    and name keys and treats a hit on either as the same company (this
    fixes a confirmed bug where querying "amazon" by name alone missed
    the existing "amazon.com" row keyed by website, creating a
    duplicate). If a name-only match is found and the current call also
    supplies a website the stored row lacks, the row is backfilled with
    that website rather than left incomplete.
"""

import json
import sqlite3
from typing import Optional


# ---------------------------------------------------------------------------
# Connection / schema setup
# ---------------------------------------------------------------------------

def init_db(path: str = "data/optimisegeo.db") -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at `path`, create all tables and
    indexes if they don't already exist, and return the connection.

    row_factory is set to sqlite3.Row so downstream functions can convert
    rows to plain dicts. Foreign keys are enabled explicitly since SQLite
    does not enforce them by default.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL UNIQUE,
            primary_count    INTEGER NOT NULL DEFAULT 0,
            secondary_count  INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS companies (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            website         TEXT,
            keywords        TEXT,
            primary_tag_id  INTEGER,
            embedding_id    TEXT,
            FOREIGN KEY (primary_tag_id) REFERENCES categories(id)
        );

        CREATE TABLE IF NOT EXISTS company_tags (
            company_id   INTEGER NOT NULL,
            category_id  INTEGER NOT NULL,
            PRIMARY KEY (company_id, category_id),
            FOREIGN KEY (company_id) REFERENCES companies(id),
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );

        CREATE INDEX IF NOT EXISTS idx_companies_website
            ON companies(website);

        CREATE INDEX IF NOT EXISTS idx_companies_name_lower
            ON companies(LOWER(name));
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_website(url: str) -> str:
    """
    Normalize a website string for exact-match comparison/storage:
    strip protocol (http:// or https://), strip a leading "www.",
    strip a single trailing slash, and lowercase the result.

    Empty/None input returns "".
    """
    if not url:
        return ""

    normalized = url.strip().lower()

    if normalized.startswith("https://"):
        normalized = normalized[len("https://"):]
    elif normalized.startswith("http://"):
        normalized = normalized[len("http://"):]

    if normalized.startswith("www."):
        normalized = normalized[len("www."):]

    if normalized.endswith("/"):
        normalized = normalized[:-1]

    return normalized


# ---------------------------------------------------------------------------
# Row <-> dict helper (decodes the JSON-encoded keywords column)
# ---------------------------------------------------------------------------

def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    """
    Convert a companies row into a plain dict with `keywords` decoded from
    JSON text back into a Python list (or [] if NULL/unparseable). Every
    read path for the companies table funnels through here so callers
    never have to think about the on-disk JSON encoding.
    """
    if row is None:
        return None
    data = dict(row)
    raw = data.get("keywords")
    if raw:
        try:
            data["keywords"] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            data["keywords"] = []
    else:
        data["keywords"] = []
    return data


# ---------------------------------------------------------------------------
# Exact-match fast path
# ---------------------------------------------------------------------------

def _find_company_by_website(conn: sqlite3.Connection, normalized_website: str) -> Optional[dict]:
    """
    Internal helper: look up a company by its normalized website. Uses the
    index on companies(website) -> O(log n) index seek, single row
    returned. `normalized_website` must already be normalized.
    """
    if not normalized_website:
        return None
    row = conn.execute(
        "SELECT * FROM companies WHERE website = ? LIMIT 1",
        (normalized_website,),
    ).fetchone()
    return row


def _find_company_by_name(conn: sqlite3.Connection, name: str) -> Optional[dict]:
    """
    Internal helper: look up a company by case-insensitive exact name
    match. Uses the index on companies(LOWER(name)) -> index seek rather
    than a full table scan, as long as the query filters on LOWER(name)
    to match the indexed expression.
    """
    if not name:
        return None
    row = conn.execute(
        "SELECT * FROM companies WHERE LOWER(name) = LOWER(?) LIMIT 1",
        (name,),
    ).fetchone()
    return row


def exact_match_lookup(conn: sqlite3.Connection, website: str = "", name: str = "") -> Optional[dict]:
    """
    THE fast-path dedup check. Checks website first (indexed seek), then
    falls back to name (indexed seek) — a hit on EITHER key is treated as
    the same company, so a call like exact_match_lookup(website="", name=
    "amazon") will still find a row that was originally inserted keyed on
    "amazon.com" via its website column.

    If the match came from the name branch and the caller supplied a
    website that the stored row doesn't have, the row is backfilled with
    that website in place (so the row becomes discoverable via either key
    from then on) rather than left permanently incomplete.

    Returns a dict with `keywords` already decoded to a list, or None if
    neither key matches anything.
    """
    normalized_website = normalize_website(website) if website else ""

    row = _find_company_by_website(conn, normalized_website)
    if row is not None:
        return _row_to_dict(row)

    row = _find_company_by_name(conn, name)
    if row is None:
        return None

    if normalized_website and not row["website"]:
        conn.execute(
            "UPDATE companies SET website = ? WHERE id = ?",
            (normalized_website, row["id"]),
        )
        conn.commit()
        row = _find_company_by_website(conn, normalized_website)

    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

def get_all_companies(conn: sqlite3.Connection) -> list[dict]:
    """Return every company row, most recently added first, with keywords
    decoded to lists. Used by the Streamlit /Data page to list all
    categorized companies."""
    rows = conn.execute("SELECT * FROM companies ORDER BY id DESC").fetchall()
    return [_row_to_dict(row) for row in rows]


def get_company_by_id(conn: sqlite3.Connection, company_id: int) -> Optional[dict]:
    """Look up a company by primary key. Used by main.py to hydrate the
    companies retrieved by embed.top_k_similar (which only returns ids)."""
    row = conn.execute(
        "SELECT * FROM companies WHERE id = ? LIMIT 1", (company_id,)
    ).fetchone()
    return _row_to_dict(row)


def update_company_embedding_id(conn: sqlite3.Connection, company_id: int, embedding_id: str) -> None:
    """
    Set companies.embedding_id after the fact. Needed because the Chroma
    embedding id is derived from the company's own row id
    (embed.embed_and_store uses f"company-{company_id}"), so the row must
    exist before the embedding id is known: insert_company first with a
    placeholder, embed, then call this to backfill the real id.
    """
    conn.execute(
        "UPDATE companies SET embedding_id = ? WHERE id = ?",
        (embedding_id, company_id),
    )
    conn.commit()


def delete_company(conn: sqlite3.Connection, company_id: int) -> Optional[str]:
    """
    Delete a company entirely: its company_tags junction rows, the
    companies row itself, and decrement the primary/secondary counts on
    whatever categories it had been tagged with (categories themselves
    are left in place even if their count hits 0 — the pool is shared
    canonical state, not owned by any one company).

    Returns the deleted row's embedding_id (or None) so the caller can
    also remove it from Chroma — that's a separate store this function
    doesn't know about.
    """
    company = get_company_by_id(conn, company_id)
    if company is None:
        return None

    if company["primary_tag_id"] is not None:
        conn.execute(
            "UPDATE categories SET primary_count = MAX(primary_count - 1, 0) WHERE id = ?",
            (company["primary_tag_id"],),
        )

    for secondary in get_secondary_tags(conn, company_id):
        conn.execute(
            "UPDATE categories SET secondary_count = MAX(secondary_count - 1, 0) WHERE id = ?",
            (secondary["id"],),
        )

    conn.execute("DELETE FROM company_tags WHERE company_id = ?", (company_id,))
    conn.execute("DELETE FROM companies WHERE id = ?", (company_id,))
    conn.commit()
    return company["embedding_id"]


def insert_company(
    conn: sqlite3.Connection,
    name: str,
    website: str,
    keywords: list[str],
    primary_tag_id: int,
    embedding_id: str,
) -> int:
    """
    Insert a new company row and return its new id.

    Takes `keywords` (a list of strings), NOT a summary — the prose
    summary is ephemeral and never reaches this module. `keywords` is
    JSON-encoded before storage (json.dumps) since SQLite has no native
    array type. `website` is normalized here defensively (idempotent if
    the caller already normalized it) so the stored value is always
    consistent with what exact_match_lookup expects.
    """
    normalized = normalize_website(website)
    encoded_keywords = json.dumps(keywords or [])
    cur = conn.execute(
        """
        INSERT INTO companies (name, website, keywords, primary_tag_id, embedding_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, normalized, encoded_keywords, primary_tag_id, embedding_id),
    )
    conn.commit()
    return cur.lastrowid


def add_secondary_tag(conn: sqlite3.Connection, company_id: int, category_id: int) -> None:
    """
    Attach a single secondary category tag to a company via the
    company_tags junction table. Uses INSERT OR IGNORE so re-running is
    idempotent (the composite primary key (company_id, category_id)
    prevents dupes). Call this once per secondary tag (0-3 per company).

    Does NOT bump categories.secondary_count itself — call
    increment_counts(conn, category_id, "secondary") for that, keeping
    "attach the tag" and "update aggregate counts" as separate, explicit
    steps.
    """
    conn.execute(
        "INSERT OR IGNORE INTO company_tags (company_id, category_id) VALUES (?, ?)",
        (company_id, category_id),
    )
    conn.commit()


def add_secondary_tags(conn: sqlite3.Connection, company_id: int, category_ids: list[int]) -> None:
    """
    Convenience wrapper around add_secondary_tag for the common case of
    attaching 0-3 secondary tags at once. Kept alongside the singular
    add_secondary_tag (rather than replacing it) since main.py already
    calls this plural form with a list of category ids.
    """
    for category_id in category_ids or []:
        add_secondary_tag(conn, company_id, category_id)


# ---------------------------------------------------------------------------
# Categories (the shared canonical tag pool)
# ---------------------------------------------------------------------------

def get_all_categories(conn: sqlite3.Connection) -> list[dict]:
    """
    Return every category as a list of dicts (id, name, primary_count,
    secondary_count). This is the full pool that gets rendered as plain
    text into the tag-assignment prompt (per CLAUDE.md: no vector search
    over tags, the list is passed in full since it stays small).
    """
    rows = conn.execute(
        "SELECT id, name, primary_count, secondary_count FROM categories ORDER BY name"
    ).fetchall()
    return [dict(row) for row in rows]


def find_category_by_id(conn: sqlite3.Connection, category_id: Optional[int]) -> Optional[dict]:
    """Look up a category by primary key. Used to resolve a company's
    primary_tag_id into a name for display/few-shot-example purposes."""
    if category_id is None:
        return None
    row = conn.execute(
        "SELECT * FROM categories WHERE id = ? LIMIT 1", (category_id,)
    ).fetchone()
    return dict(row) if row else None


def get_secondary_tags(conn: sqlite3.Connection, company_id: int) -> list[dict]:
    """
    Return the secondary categories attached to a company via the
    company_tags junction table, as a list of category dicts.
    """
    rows = conn.execute(
        """
        SELECT categories.* FROM categories
        JOIN company_tags ON company_tags.category_id = categories.id
        WHERE company_tags.company_id = ?
        ORDER BY categories.name
        """,
        (company_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def find_category_by_name(conn: sqlite3.Connection, name: str) -> Optional[dict]:
    """
    Case-insensitive exact-name lookup of a single category. This is the
    exact-match step of the string-level dedup guardrail (normalize ->
    exact match check -> rapidfuzz fallback lives in the caller, not here).
    """
    if not name:
        return None
    row = conn.execute(
        "SELECT * FROM categories WHERE LOWER(name) = LOWER(?) LIMIT 1",
        (name,),
    ).fetchone()
    return dict(row) if row else None


def insert_category(conn: sqlite3.Connection, name: str) -> int:
    """
    Insert a brand-new category with zeroed counts and return its id.
    Relies on the UNIQUE constraint on categories.name as a last-resort
    safety net; callers are expected to have already run the string-level
    dedup guardrail (exact match + rapidfuzz token_set_ratio >= 88) before
    calling this, per CLAUDE.md.
    """
    cur = conn.execute(
        "INSERT INTO categories (name, primary_count, secondary_count) VALUES (?, 0, 0)",
        (name,),
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_category(conn: sqlite3.Connection, name: str) -> int:
    """
    Convenience wrapper: return the id of the category matching `name`
    (case-insensitive exact match), creating it if it doesn't exist yet.
    """
    existing = find_category_by_name(conn, name)
    if existing:
        return existing["id"]
    return insert_category(conn, name)


def increment_counts(conn: sqlite3.Connection, category_id: int, kind: str) -> None:
    """
    Increment a category's usage counter. `kind` must be "primary" or
    "secondary" and selects which counter column gets bumped by 1.

    These counts are pure bookkeeping (e.g. for reporting which
    categories are most-used) and are kept separate from the
    company_tags junction table, which is the source of truth for
    which companies have which secondary tags.
    """
    if kind not in ("primary", "secondary"):
        raise ValueError(f'kind must be "primary" or "secondary", got {kind!r}')

    column = "primary_count" if kind == "primary" else "secondary_count"
    conn.execute(
        f"UPDATE categories SET {column} = {column} + 1 WHERE id = ?",
        (category_id,),
    )
    conn.commit()


# Backwards-compatible alias: main.py currently calls db.increment_tag_count(...).
# Kept as a thin alias (not a duplicated implementation) rather than renaming
# every call site, since main.py is out of scope for this change.
increment_tag_count = increment_counts
