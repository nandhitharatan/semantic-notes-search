"""Display ingestion status across PostgreSQL and Qdrant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "notes_chunks"
load_dotenv(PROJECT_ROOT / ".env")


def _postgres_connection():
    """Connect through .env so the status report reads the durable relational store."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _qdrant_client() -> QdrantClient:
    """Connect through .env so the report counts points in the configured vector index."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing required environment variable: QDRANT_URL")
    return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)


def _postgres_status() -> tuple[list[dict[str, Any]], int]:
    """Read document rows and independent chunk/job aggregates without changing database state."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.doc_id,
                    d.file_name,
                    d.subject,
                    COALESCE(c.chunk_count, 0) AS chunk_count,
                    COALESCE(j.status, 'NO JOB') AS job_status
                FROM documents AS d
                LEFT JOIN (
                    SELECT doc_id, COUNT(*) AS chunk_count
                    FROM chunks
                    GROUP BY doc_id
                ) AS c ON c.doc_id = d.doc_id
                LEFT JOIN (
                    SELECT DISTINCT ON (doc_id) doc_id, status
                    FROM ingestion_jobs
                    ORDER BY doc_id, job_id DESC
                ) AS j ON j.doc_id = d.doc_id
                ORDER BY d.doc_id
                """
            )
            rows = [
                {
                    "doc_id": row[0],
                    "file_name": row[1],
                    "subject": row[2] or "",
                    "chunk_count": row[3],
                    "job_status": row[4],
                }
                for row in cursor.fetchall()
            ]

            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
    finally:
        connection.close()

    return rows, total_chunks


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print one readable row per document for a viva-friendly consistency report."""
    headers = ("doc_id", "file_name", "subject", "chunks", "job_status")
    values = [
        (
            str(row["doc_id"]),
            row["file_name"],
            row["subject"],
            str(row["chunk_count"]),
            row["job_status"],
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(value[index]) for value in values)) if values else len(header)
        for index, header in enumerate(headers)
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for value in values:
        print(format_row(value))


def main() -> None:
    """Print per-document PostgreSQL status followed by total PostgreSQL and Qdrant counts."""
    rows, total_chunks = _postgres_status()
    _print_table(rows)
    points_count = _qdrant_client().count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count
    print(f"Total chunks in PostgreSQL: {total_chunks}")
    print(f"points_count in Qdrant: {points_count}")


if __name__ == "__main__":
    main()
