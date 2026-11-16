"""Delete only the two approved test documents from PostgreSQL and Qdrant."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_NAME = "notes_chunks"
TARGET_DOCUMENTS = {1: "DBMS", 2: "FAILTEST"}
load_dotenv(PROJECT_ROOT / ".env")


def _postgres_connection():
    """Connect through .env so the cleanup removes rows only from the configured source database."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _qdrant_client() -> QdrantClient:
    """Connect through .env so the cleanup removes matching vector points by stable chunk UUID."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing required environment variable: QDRANT_URL")
    return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)


def _counts() -> tuple[int, int]:
    """Read total PostgreSQL chunks and Qdrant points to make cleanup effects observable."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            postgres_chunks = cursor.fetchone()[0]
    finally:
        connection.close()

    qdrant_points = _qdrant_client().count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count
    return postgres_chunks, qdrant_points


def _target_chunk_ids() -> dict[int, list[str] | None]:
    """Verify approved documents and read their chunk UUIDs, tolerating already-cleaned targets."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT doc_id, subject FROM documents WHERE doc_id = ANY(%s)",
                (list(TARGET_DOCUMENTS),),
            )
            found_subjects = {row[0]: row[1] for row in cursor.fetchall()}
            unexpected_subjects = {
                doc_id: subject
                for doc_id, subject in found_subjects.items()
                if subject != TARGET_DOCUMENTS[doc_id]
            }
            if unexpected_subjects:
                raise RuntimeError(
                    f"Target verification failed for subject(s): {unexpected_subjects}."
                )

            chunk_ids: dict[int, list[str] | None] = {}
            for doc_id in TARGET_DOCUMENTS:
                if doc_id not in found_subjects:
                    chunk_ids[doc_id] = None
                    continue
                cursor.execute(
                    "SELECT chunk_id::text FROM chunks WHERE doc_id = %s ORDER BY chunk_index",
                    (doc_id,),
                )
                chunk_ids[doc_id] = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()
    return chunk_ids


def _delete_qdrant_points(client: QdrantClient, chunk_ids: list[str]) -> int:
    """Retrieve before deleting so points already removed by an earlier run are treated as success."""
    if not chunk_ids:
        return 0
    existing_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=chunk_ids,
        with_payload=False,
        with_vectors=False,
    )
    existing_ids = [point.id for point in existing_points]
    if existing_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=existing_ids),
            wait=True,
        )
    return len(existing_ids)


def _delete_document_rows(doc_id: int) -> tuple[int, int]:
    """Delete only one approved document after its Qdrant points are removed first."""
    connection = _postgres_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                # ingestion_jobs has no ON DELETE CASCADE in the existing schema.
                cursor.execute("DELETE FROM ingestion_jobs WHERE doc_id = %s", (doc_id,))
                jobs_deleted = cursor.rowcount
                cursor.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
                documents_deleted = cursor.rowcount
    finally:
        connection.close()
    return jobs_deleted, documents_deleted


def main() -> None:
    """Show counts, require YES, then clean Qdrant and PostgreSQL for documents 1 and 2 only."""
    postgres_before, qdrant_before = _counts()
    chunk_ids_by_doc = _target_chunk_ids()

    print(f"Before: PostgreSQL chunks={postgres_before}, Qdrant points={qdrant_before}")
    for doc_id, subject in TARGET_DOCUMENTS.items():
        chunk_ids = chunk_ids_by_doc[doc_id]
        state = "already absent" if chunk_ids is None else f"chunk_ids={len(chunk_ids)}"
        print(f"Target doc_id={doc_id}, subject={subject}, {state}")

    confirmation = input("Type YES to delete only document IDs 1 and 2: ")
    if confirmation != "YES":
        print("Cancelled. No Qdrant or PostgreSQL data was deleted.")
        return

    client = _qdrant_client()
    for doc_id, chunk_ids in chunk_ids_by_doc.items():
        if chunk_ids is None:
            print(f"Skipped doc_id={doc_id}: already absent")
            continue
        qdrant_deleted = _delete_qdrant_points(client, chunk_ids)
        jobs_deleted, documents_deleted = _delete_document_rows(doc_id)
        print(
            f"Deleted doc_id={doc_id}: Qdrant point IDs={qdrant_deleted}, "
            f"PostgreSQL jobs={jobs_deleted}, documents={documents_deleted}"
        )

    postgres_after, qdrant_after = _counts()
    print(f"After: PostgreSQL chunks={postgres_after}, Qdrant points={qdrant_after}")


if __name__ == "__main__":
    main()
