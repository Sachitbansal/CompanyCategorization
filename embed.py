"""
embed.py — company summary embeddings for optimiseGEO.

Only company summaries get embedded (sentence-transformers all-MiniLM-L6-v2
-> Chroma), never the tag list itself — the category pool stays small
enough to pass as plain text in the tag-assignment prompt (see
prompts/tagging_prompt.py), per CLAUDE.md. Embeddings exist purely to
retrieve top-k similar companies as few-shot examples for that prompt.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "company_summaries"
CHROMA_PATH = "chroma_db"


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


def embed_and_store(
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    company_id: int,
    summary: str,
) -> str:
    """
    Embed `summary` and store it in the collection, keyed by an id derived
    from `company_id`. Returns that embedding_id so the caller can store
    it on the companies row (companies.embedding_id) for later lookup.
    """
    embedding_id = f"company-{company_id}"
    vector = embedder.encode(summary).tolist()
    collection.upsert(
        ids=[embedding_id],
        embeddings=[vector],
        documents=[summary],
        metadatas=[{"company_id": company_id}],
    )
    return embedding_id


def top_k_similar(
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    summary: str,
    k: int = 5,
    threshold: float = 0.35,
) -> list[dict]:
    """
    Retrieve up to k companies whose stored summaries are most similar to
    `summary`. Returns a list of dicts: {company_id, summary, similarity}.

    Collection distance space is cosine (0 = identical, 2 = opposite), so
    similarity = 1 - distance. Only matches with similarity >= threshold
    are returned; if NONE clear the threshold the list is empty, and per
    CLAUDE.md the caller should send the tag-assignment prompt with no
    few-shot examples rather than forcing in weak/irrelevant ones.

    threshold=0.35, not the originally-planned 0.60: measured on real
    company summaries, all-MiniLM-L6-v2 cosine similarity for genuinely
    similar businesses (e.g. Amazon vs eBay, both marketplaces) comes out
    around 0.46, and unrelated companies are usually under 0.20. A 0.60
    cutoff would reject every real match seen so far. CLAUDE.md flags
    this threshold as "tunable" for exactly this reason.
    """
    if collection.count() == 0:
        return []

    vector = embedder.encode(summary).tolist()
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
                    "summary": document,
                    "similarity": similarity,
                }
            )

    return matches
