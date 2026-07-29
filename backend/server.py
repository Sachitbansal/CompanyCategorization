"""
server.py — FastAPI wrapper around the Brand Categoriser pipeline.

Two things live here: a WebSocket endpoint that streams run_pipeline's
step events live to the frontend (replacing Streamlit's st.status), and a
small REST API for the Data page (list/delete companies).

run_pipeline itself (main.py) is unchanged — this file is pure plumbing,
it does not touch pipeline logic.

IMPORTANT: run_pipeline is a SYNCHRONOUS generator that does blocking work
between yields (HTTP scraping, DeepSeek calls, sentence-transformers
encoding). It must NOT be iterated directly inside the async WebSocket
handler: `await websocket.send_json(...)` only hands the message to the
ASGI layer, and the actual socket write happens back on the event loop —
which stays starved for as long as the blocking work runs. The visible
symptom is every log line arriving in one burst at the end instead of
streaming. So the generator runs on a worker thread and its events are
pumped back to the event loop through an asyncio.Queue (see
_stream_pipeline), which keeps the loop free to flush each send as it
happens.
"""

import asyncio
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import db
import embed
from main import run_pipeline

app = FastAPI(title="Brand Categoriser API")

# Single-user local/dev tool — wide open CORS is fine here. Tighten
# allow_origins to the actual frontend URL before exposing this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serialize_event(event: dict) -> dict:
    """
    run_pipeline's "awaiting_context" event carries a `scraped`
    ScrapedContent dataclass instance for the server to hold onto across
    the pause — that's not JSON-serializable and the frontend has no use
    for it (it only needs to show the draft summary and collect the
    user's description), so it's stripped before sending.
    """
    return {k: v for k, v in event.items() if k != "scraped"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/companies")
def list_companies():
    """One row per company, with its primary/secondary tags resolved and
    keywords decoded — everything the Data page needs in one call."""
    conn = db.init_db()
    companies = db.get_all_companies(conn)
    result = []
    for company in companies:
        primary = db.find_category_by_id(conn, company["primary_tag_id"])
        secondaries = db.get_secondary_tags(conn, company["id"])
        result.append(
            {
                "id": company["id"],
                "name": company["name"],
                "website": company["website"],
                "keywords": company["keywords"],
                "primary_tag": primary["name"] if primary else None,
                "secondary_tags": [s["name"] for s in secondaries],
            }
        )
    return result


@app.delete("/api/companies/{company_id}")
def delete_company(company_id: int):
    """Removes the company from both SQLite (row, junction rows, category
    counts) and Chroma (its embedding) — same two-store cleanup the
    Streamlit Data page used to do directly."""
    conn = db.init_db()
    embedding_id = db.delete_company(conn, company_id)
    if embedding_id:
        collection = embed.get_chroma_collection()
        embed.delete_embedding(collection, embedding_id)
    return {"deleted": company_id}


_DONE = object()  # sentinel: worker thread finished producing events


async def _stream_pipeline(websocket: WebSocket, name_or_url: str, extra_context, scraped):
    """
    Run one pass of the blocking `run_pipeline` generator on a worker
    thread, forwarding each event to the socket the moment it's produced.

    The thread pushes events onto an asyncio.Queue via
    loop.call_soon_threadsafe (the only thread-safe way to touch loop
    state from outside it), while this coroutine sits in `await
    queue.get()` — so the event loop stays idle and free to actually
    flush each send_json to the network between pipeline steps. That's
    what makes the log stream live instead of arriving as one burst at
    the end.

    Returns the "awaiting_context" event if the run paused for user
    input, else None.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for event in run_pipeline(name_or_url, extra_context=extra_context, scraped=scraped):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # noqa: BLE001 — surface pipeline failures to the client instead of dropping the socket
            loop.call_soon_threadsafe(
                queue.put_nowait, {"step": "error", "status": "done", "detail": str(exc)}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    threading.Thread(target=worker, daemon=True).start()

    pending = None
    while True:
        event = await queue.get()
        if event is _DONE:
            return pending
        if event["step"] == "awaiting_context":
            pending = event
        await websocket.send_json(_serialize_event(event))


@app.websocket("/ws/categorize")
async def categorize_ws(websocket: WebSocket):
    """
    Protocol: client sends {"name_or_url": "..."}. Server streams back one
    JSON message per run_pipeline event. If a run pauses on
    {"step": "awaiting_context", ...}, the client must reply on the SAME
    connection with {"extra_context": "<description or empty string>"} to
    resume — the server keeps the already-fetched `scraped` object in a
    local variable across that round-trip so the site isn't re-scraped.
    After a run reaches "complete" (or errors), the connection stays open
    and waits for the next {"name_or_url": ...} message, so one page
    session can categorize multiple brands over one socket.
    """
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            name_or_url = (payload.get("name_or_url") or "").strip()
            if not name_or_url:
                await websocket.send_json({"step": "error", "status": "done", "detail": "empty input"})
                continue

            extra_context = None
            scraped = None

            while True:
                pending = await _stream_pipeline(websocket, name_or_url, extra_context, scraped)

                if pending is None:
                    break

                scraped = pending["scraped"]
                resume = await websocket.receive_json()
                extra_context = resume.get("extra_context") or ""
    except WebSocketDisconnect:
        pass
