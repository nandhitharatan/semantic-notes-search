"""PDF ingestion into PostgreSQL and Qdrant for the Semantic Notes Search Engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "notes_chunks"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# All connection details are loaded from the local, gitignored .env file.
load_dotenv(PROJECT_ROOT / ".env")


def _required_setting(name: str) -> str:
    """Read a required connection setting so credentials stay outside source control."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _split_text(text: str) -> list[str]:
    """Create overlapping word-boundary chunks so indexed units preserve border context."""
    normalised_text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(normalised_text):
        if len(normalised_text) - start <= CHUNK_SIZE:
            chunks.append(normalised_text[start:].strip())
            break

        boundary = normalised_text.rfind(" ", start + 1, start + CHUNK_SIZE + 1)
        if boundary <= start:
            raise ValueError("A word is longer than the 500-character chunk limit.")

        chunks.append(normalised_text[start:boundary].strip())

        overlap_target = max(start + 1, boundary - CHUNK_OVERLAP)
        before_target = normalised_text.rfind(" ", start, overlap_target + 1)
        after_target = normalised_text.find(" ", overlap_target, boundary)
        candidates = [
            position
            for position in (before_target, after_target)
            if start < position < boundary
        ]
        if not candidates:
            raise ValueError("Could not find a word boundary for chunk overlap.")
        overlap_boundary = min(candidates, key=lambda position: abs(position - overlap_target))
        start = overlap_boundary + 1

    return [chunk for chunk in chunks if chunk]


def extract_pdf_chunks(path: str | Path) -> list[tuple[int, str]]:
    """Extract page-tagged chunks so search results retain relational provenance to a PDF page."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    page_chunks: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_chunks.extend((page_no, chunk) for chunk in _split_text(page_text))

    if not page_chunks:
        raise ValueError("The PDF contains no extractable text.")
    return page_chunks


def _insert_postgres_records(
    file_name: str,
    subject: str,
    user_id: int,
    page_chunks: list[tuple[int, str]],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Commit document, chunk, and PENDING-job rows first for durable outbox-style ingestion."""
    chunk_records = [
        {
            "chunk_id": str(uuid4()),
            "chunk_index": chunk_index,
            "page_no": page_no,
            "text": text,
            "token_count": len(text.split()),
        }
        for chunk_index, (page_no, text) in enumerate(page_chunks)
    ]

    connection = psycopg2.connect(_required_setting("POSTGRES_URL"))
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO documents (user_id, file_name, subject)
                    VALUES (%s, %s, %s)
                    RETURNING doc_id
                    """,
                    (user_id, file_name, subject),
                )
                doc_id = cursor.fetchone()[0]

                execute_values(
                    cursor,
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, chunk_index, chunk_text, token_count, page_no)
                    VALUES %s
                    """,
                    [
                        (
                            record["chunk_id"],
                            doc_id,
                            record["chunk_index"],
                            record["text"],
                            record["token_count"],
                            record["page_no"],
                        )
                        for record in chunk_records
                    ],
                )

                cursor.execute(
                    """
                    INSERT INTO ingestion_jobs (doc_id, status)
                    VALUES (%s, 'PENDING')
                    RETURNING job_id
                    """,
                    (doc_id,),
                )
                job_id = cursor.fetchone()[0]
    finally:
        connection.close()

    return doc_id, job_id, chunk_records


def _qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment configuration to keep infrastructure swappable."""
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=_required_setting("QDRANT_URL"), api_key=api_key)


def _upsert_embeddings(
    doc_id: int, subject: str, chunk_records: list[dict[str, Any]]
) -> None:
    """Upsert vectors with chunk UUIDs so retries are idempotent rather than creating duplicates."""
    client = _qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
            "Run database/qdrant_setup.py first."
        )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    for start in range(0, len(chunk_records), EMBEDDING_BATCH_SIZE):
        batch = chunk_records[start : start + EMBEDDING_BATCH_SIZE]
        vectors = model.encode(
            [record["text"] for record in batch],
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        points = [
            PointStruct(
                id=record["chunk_id"],
                vector=vector.tolist(),
                payload={
                    "doc_id": doc_id,
                    "subject": subject,
                    "chunk_index": record["chunk_index"],
                    "text": record["text"],
                },
            )
            for record, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


def _mark_job_completed(job_id: int) -> None:
    """Mark completion only after Qdrant succeeds, preserving consistency repair work when it does not."""
    connection = psycopg2.connect(_required_setting("POSTGRES_URL"))
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'COMPLETED', error_message = NULL
                    WHERE job_id = %s
                    """,
                    (job_id,),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"Ingestion job {job_id} was not found.")
    finally:
        connection.close()


def _job_status(job_id: int) -> str:
    """Read the durable job state so the CLI can show PostgreSQL's consistency record."""
    connection = psycopg2.connect(_required_setting("POSTGRES_URL"))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status FROM ingestion_jobs WHERE job_id = %s", (job_id,)
            )
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError(f"Ingestion job {job_id} was not found.")
    return row[0]


def _qdrant_point_count(doc_id: int) -> int:
    """Count a document's vectors to verify its PostgreSQL chunks reached the vector index."""
    result = _qdrant_client().count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
        exact=True,
    )
    return result.count


def ingest_pdf(path: str | Path, subject: str, user_id: int) -> dict[str, int | str]:
    """Ingest one PDF using PostgreSQL as the durable source of truth and Qdrant as its vector index."""
    if not subject.strip():
        raise ValueError("subject must not be empty")

    pdf_path = Path(path)
    page_chunks = extract_pdf_chunks(pdf_path)
    doc_id, job_id, chunk_records = _insert_postgres_records(
        file_name=pdf_path.name,
        subject=subject.strip(),
        user_id=user_id,
        page_chunks=page_chunks,
    )

    try:
        _upsert_embeddings(doc_id, subject.strip(), chunk_records)
    except Exception as error:
        # PostgreSQL was committed above; the PENDING job is intentionally left for Step 4 retries.
        raise RuntimeError(
            f"PostgreSQL rows were saved for document {doc_id}, but Qdrant ingestion failed. "
            f"Job {job_id} remains PENDING."
        ) from error

    _mark_job_completed(job_id)
    return {
        "doc_id": doc_id,
        "job_id": job_id,
        "chunk_count": len(chunk_records),
        "job_status": _job_status(job_id),
        "qdrant_point_count": _qdrant_point_count(doc_id),
    }


def main() -> None:
    """Run one ingestion from the command line to make the two-database result observable."""
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: py src/ingest.py <pdf_path> <subject> <user_id>"
        )

    result = ingest_pdf(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    print(
        "Ingestion complete: "
        f"chunks={result['chunk_count']}, "
        f"job_status={result['job_status']}, "
        f"qdrant_points={result['qdrant_point_count']}"
    )


if __name__ == "__main__":
    main()
