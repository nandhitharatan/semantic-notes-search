"""PostgreSQL search methods for the Semantic Notes Search Engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _connection():
    """Open PostgreSQL from .env so both searches use the durable relational database."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _validate_limit(limit: int) -> None:
    """Bound result counts so a search cannot accidentally request an unbounded result set."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")


def search_like(query: str, limit: int) -> list[dict[str, Any]]:
    """Use ILIKE for a simple substring baseline against PostgreSQL's stored chunk text."""
    _validate_limit(limit)
    connection = _connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    d.file_name,
                    d.subject,
                    c.page_no,
                    c.chunk_index,
                    c.chunk_text
                FROM chunks AS c
                JOIN documents AS d ON d.doc_id = c.doc_id
                WHERE c.chunk_text ILIKE %s
                ORDER BY c.chunk_id
                LIMIT %s
                """,
                (f"%{query}%", limit),
            )
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def search_fts(query: str, limit: int) -> list[dict[str, Any]]:
    """Use the GIN-backed English tsvector query and ts_rank to demonstrate indexed FTS retrieval."""
    _validate_limit(limit)
    connection = _connection()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    c.chunk_id,
                    c.doc_id,
                    d.file_name,
                    d.subject,
                    c.page_no,
                    c.chunk_index,
                    c.chunk_text,
                    ts_rank(
                        to_tsvector('english', c.chunk_text),
                        plainto_tsquery('english', %s)
                    ) AS rank
                FROM chunks AS c
                JOIN documents AS d ON d.doc_id = c.doc_id
                WHERE to_tsvector('english', c.chunk_text)
                    @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC, c.chunk_id
                LIMIT %s
                """,
                (query, query, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
