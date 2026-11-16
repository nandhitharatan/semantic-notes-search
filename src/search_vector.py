"""Qdrant semantic search for the Semantic Notes Search Engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "notes_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
load_dotenv(PROJECT_ROOT / ".env")


def _qdrant_client() -> QdrantClient:
    """Create the Qdrant connection from .env so vector storage remains configurable."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing required environment variable: QDRANT_URL")
    return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)


def _validate_inputs(limit: int, hnsw_ef: int | None) -> None:
    """Validate query controls so benchmark comparisons use explicit, positive parameters."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if hnsw_ef is not None and hnsw_ef <= 0:
        raise ValueError("hnsw_ef must be greater than zero")


def search_vector(
    query: str,
    limit: int,
    subject: str | None = None,
    exact: bool = False,
    hnsw_ef: int | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant with cosine embeddings, optional subject filtering, and explicit HNSW controls."""
    if not query.strip():
        raise ValueError("query must not be empty")
    _validate_inputs(limit, hnsw_ef)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vector = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    query_filter = None
    if subject:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="subject",
                    match=MatchValue(value=subject),
                )
            ]
        )

    response = _qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        search_params=SearchParams(hnsw_ef=hnsw_ef, exact=exact),
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "text": (point.payload or {}).get("text", ""),
            "score": point.score,
            "doc_id": (point.payload or {}).get("doc_id"),
        }
        for point in response.points
    ]
