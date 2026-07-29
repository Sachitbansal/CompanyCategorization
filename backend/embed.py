"""
embed.py — company keyword embeddings for Brand Categoriser.

Only the `keywords` list gets embedded (sentence-transformers
all-MiniLM-L6-v2 -> Chroma), never the prose summary and never the tag
list itself — the category pool stays small enough to pass as plain text
in the tag-assignment prompt (see prompts/tagging_prompt.py), per
CLAUDE.md. Embeddings exist purely to retrieve top-k similar companies as
few-shot examples for that prompt.

Prose summaries were embedded here originally, but were dropped per the
tag-canonicalization skill (frontmatter name similarity-retrieval) after a
confirmed bug: full-sentence summaries matched on shared boilerplate
phrasing ("Indian", "businesses of all sizes") rather than actual
business similarity — Razorpay outranked Flipkart for a Flipkart query,
and Amazon/Flipkart (same industry) scored lower than Amazon/Razorpay
(different industries). Keywords are industry/product terms only, with
the boilerplate already stripped out at generation time
(prompts/summary_prompt.py), so they embed on signal instead of noise.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "company_keywords"
CHROMA_PATH = "data/chroma_db"


def get_embedder() -> SentenceTransformer:
    """Load the sentence-transformers model used for all summary embeddings."""
    return SentenceTransformer(EMBEDDING_MODEL)


def get_chroma_collection(path: str = CHROMA_PATH) -> chromadb.Collection:
    """
    Open (or create) the persistent Chroma collection that stores company
    summary embeddings. Persistent client so the index survives across
    runs, same as the SQLite file.

    metadata={"hnsw:space": "cosine"} is required at creation time —
    Chroma's actual default is squared-L2, not cosine, and the space
    can't be changed after a collection exists. top_k_similar's
    `1 - distance` similarity math is only correct once this is cosine.
    """
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def delete_embedding(collection: chromadb.Collection, embedding_id: str) -> None:
    """Remove a stored embedding, e.g. when its company is deleted from db.py."""
    if embedding_id:
        collection.delete(ids=[embedding_id])


def _join_keywords(keywords: list[str]) -> str:
    """Flatten a keyword list into the single string that actually gets
    embedded. ", "-joined rather than concatenated so MiniLM still tokenizes
    each keyword as a separate word/phrase rather than running them together."""
    return ", ".join(keywords)


def embed_and_store(
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    company_id: int,
    keywords: list[str],
) -> str:
    """
    Embed `keywords` and store it in the collection, keyed by an id derived
    from `company_id`. Returns that embedding_id so the caller can store
    it on the companies row (companies.embedding_id) for later lookup.
    """
    embedding_id = f"company-{company_id}"
    text = _join_keywords(keywords)
    vector = embedder.encode(text).tolist()
    collection.upsert(
        ids=[embedding_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"company_id": company_id}],
    )
    return embedding_id


def top_k_similar(
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    keywords: list[str],
    k: int = 5,
    threshold: float = 0.60,
) -> list[dict]:
    """
    Retrieve up to k companies whose stored keyword lists are most similar
    to `keywords`. Returns a list of dicts: {company_id, keywords, similarity}.

    Collection distance space is cosine (0 = identical, 2 = opposite), so
    similarity = 1 - distance. Only matches with similarity >= threshold
    are returned; if NONE clear the threshold the list is empty, and per
    CLAUDE.md the caller should send the tag-assignment prompt with no
    few-shot examples rather than forcing in weak/irrelevant ones.

    threshold=0.60 — a compromise arrived at over several rounds of
    measurement. History: 0.60 initial -> 0.35 (after fixing an
    L2-vs-cosine units bug) -> 0.65 (after switching from prose-summary to
    keyword embeddings, which shifted the whole range up) -> 0.55 (because
    HP/Dell, genuinely similar, scored only 0.584) -> 0.60 now, because
    0.55 was letting marginal cross-industry matches through and they
    actively distorted results: Zomato (food delivery) retrieved Flipkart
    at 0.569 and Alibaba at 0.604 as "similar", and those e-commerce
    examples pulled its primary tag to "E-commerce" instead of a food
    sector. 0.60 cuts the weakest of those while keeping same-industry
    pairs. Note this sacrifices HP/Dell (0.584) — accepted, since a missing
    few-shot example is far cheaper than a wrong one (with none, the model
    reasons from the company's own keywords, which is the safer failure
    mode). Keep re-measuring as the company set grows.
    """
    if collection.count() == 0:
        return []

    text = _join_keywords(keywords)
    vector = embedder.encode(text).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(k, collection.count()),
    )

    matches = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for doc_id, document, distance, metadata in zip(ids, documents, distances, metadatas):
        similarity = 1 - distance
        if similarity >= threshold:
            matches.append(
                {
                    "company_id": metadata["company_id"],
                    "keywords": document.split(", ") if document else [],
                    "similarity": similarity,
                }
            )

    return matches
