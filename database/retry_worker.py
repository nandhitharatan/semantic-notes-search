"""Retry PENDING PostgreSQL ingestion jobs and repair their Qdrant vector index entries."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "notes_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
MAX_RETRIES = 3
load_dotenv(PROJECT_ROOT / ".env")


def _postgres_connection():
    """Connect through .env so the worker reads the durable outbox and source chunks."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _qdrant_client() -> QdrantClient:
    """Connect through .env so retry delivery uses the configured vector index."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing required environment variable: QDRANT_URL")
    return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)


def _mark_exhausted_jobs() -> int:
    """Mark already-exhausted outbox rows FAILED so they are not retried indefinitely."""
    connection = _postgres_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ingestion_jobs
                    SET status = 'FAILED',
                        error_message = 'Maximum retry count reached'
                    WHERE status = 'PENDING' AND retry_count >= %s
                    """,
                    (MAX_RETRIES,),
                )
                return cursor.rowcount
    finally:
        connection.close()


def _pending_jobs() -> list[dict[str, Any]]:
    """Read retryable PENDING jobs from PostgreSQL, the source of truth for consistency repair."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT j.job_id, j.doc_id, j.retry_count, d.subject
                FROM ingestion_jobs AS j
                JOIN documents AS d ON d.doc_id = j.doc_id
                WHERE j.status = 'PENDING' AND j.retry_count < %s
                ORDER BY j.job_id
                """,
                (MAX_RETRIES,),
            )
            return [
                {
                    "job_id": row[0],
                    "doc_id": row[1],
                    "retry_count": row[2],
                    "subject": row[3],
                }
                for row in cursor.fetchall()
            ]
    finally:
        connection.close()


def _job_chunks(doc_id: int) -> list[dict[str, Any]]:
    """Re-read chunks from PostgreSQL rather than memory so every retry is durable and repeatable."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id::text, chunk_index, chunk_text
                FROM chunks
                WHERE doc_id = %s
                ORDER BY chunk_index
                """,
                (doc_id,),
            )
            chunks = [
                {"chunk_id": row[0], "chunk_index": row[1], "text": row[2]}
                for row in cursor.fetchall()
            ]
    finally:
        connection.close()

    if not chunks:
        raise RuntimeError(f"Document {doc_id} has no chunks to retry.")
    return chunks


def _embed_chunks(
    model: SentenceTransformer, chunks: list[dict[str, Any]]
) -> tuple[list[list[float]], float]:
    """Embed chunks in batches with one reused model and measure only embedding time."""
    started = time.perf_counter()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        encoded_batch = model.encode(
            [chunk["text"] for chunk in batch],
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vectors.extend(vector.tolist() for vector in encoded_batch)
    embedding_ms = (time.perf_counter() - started) * 1000
    return vectors, embedding_ms


def _upsert_vectors(
    client: QdrantClient,
    job: dict[str, Any],
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> float:
    """Upsert stable chunk UUIDs and measure only Qdrant delivery time for idempotent retries."""
    started = time.perf_counter()
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = vectors[start : start + EMBEDDING_BATCH_SIZE]
        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=vector,
                payload={
                    "doc_id": job["doc_id"],
                    "subject": job["subject"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                },
            )
            for chunk, vector in zip(batch, batch_vectors, strict=True)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return (time.perf_counter() - started) * 1000


def _mark_completed(job_id: int) -> None:
    """Mark a job COMPLETED only after all of its Qdrant points were accepted."""
    connection = _postgres_connection()
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
    finally:
        connection.close()


def _record_failure(job: dict[str, Any], error: Exception) -> tuple[str, int]:
    """Increase the retry count and mark FAILED on the third failed consistency repair attempt."""
    retry_count = job["retry_count"] + 1
    status = "FAILED" if retry_count >= MAX_RETRIES else "PENDING"
    connection = _postgres_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ingestion_jobs
                    SET retry_count = %s, status = %s, error_message = %s
                    WHERE job_id = %s
                    """,
                    (retry_count, status, str(error), job["job_id"]),
                )
    finally:
        connection.close()
    return status, retry_count


def run_worker() -> None:
    """Retry all PENDING jobs once, reusing one model and printing separate embedding/upsert timings."""
    exhausted_jobs = _mark_exhausted_jobs()
    if exhausted_jobs:
        print(f"Marked {exhausted_jobs} exhausted job(s) FAILED.")

    jobs = _pending_jobs()
    if not jobs:
        print("No retryable PENDING jobs found.")
        return

    # The model is loaded once for the entire worker run, then reused for every job.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = _qdrant_client()
    print(f"Loaded one embedding model for {len(jobs)} PENDING job(s).")

    for job in jobs:
        try:
            chunks = _job_chunks(job["doc_id"])
            vectors, embedding_ms = _embed_chunks(model, chunks)
            upsert_ms = _upsert_vectors(client, job, chunks, vectors)
            _mark_completed(job["job_id"])
            print(
                f"Job {job['job_id']} COMPLETED: {len(chunks)} chunks, "
                f"embedding={embedding_ms:.2f} ms, qdrant_upsert={upsert_ms:.2f} ms"
            )
        except Exception as error:
            status, retry_count = _record_failure(job, error)
            print(
                f"Job {job['job_id']} {status}: retry_count={retry_count}, "
                f"reason={error}"
            )


if __name__ == "__main__":
    run_worker()
