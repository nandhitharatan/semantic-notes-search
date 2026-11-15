"""PDF ingestion into PostgreSQL and Qdrant for the Semantic Notes Search Engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "notes_chunks"
CHUNK_SIZE = 500
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
    """Create fixed-size chunks, the unit PostgreSQL indexes and Qdrant stores separately."""
    normalised_text = " ".join(text.split())
    return [
        normalised_text[start : start + CHUNK_SIZE]
        for start in range(0, len(normalised_text), CHUNK_SIZE)
        if normalised_text[start : start + CHUNK_SIZE]
    ]


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


def ingest_pdf(path: str | Path, subject: str, user_id: int) -> dict[str, int]:
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
    return {"doc_id": doc_id, "job_id": job_id, "chunk_count": len(chunk_records)}
