"""Manually prove PostgreSQL keeps a PENDING ingestion job when Qdrant is down."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Running this file directly makes database/ the first import path.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import COLLECTION_NAME, ingest_pdf


FAILURE_SUBJECT = "FAILTEST"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
load_dotenv(PROJECT_ROOT / ".env")


def _postgres_connection():
    """Connect through .env so the test reads the same durable PostgreSQL database as ingestion."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _qdrant_client() -> QdrantClient:
    """Connect through .env so cleanup removes test points from the configured vector store."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing required environment variable: QDRANT_URL")
    return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)


def _qdrant_service_name() -> str:
    """Read docker-compose.yml to confirm this proof test targets its Qdrant service."""
    compose_text = COMPOSE_FILE.read_text(encoding="utf-8")
    if not re.search(r"^\s{2}qdrant:\s*$", compose_text, flags=re.MULTILINE):
        raise RuntimeError("Could not find the qdrant service in docker-compose.yml")
    return "qdrant"


def qdrant_container_name() -> str:
    """Resolve Docker Compose's live container name so stop and start commands are exact."""
    service_name = _qdrant_service_name()
    result = subprocess.run(
        ["docker", "compose", "ps", service_name, "--format", "{{.Name}}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    container_name = result.stdout.strip()
    if not container_name:
        raise RuntimeError("Qdrant container is not running; start it before beginning this test.")
    return container_name


def _latest_failure_document(file_name: str) -> dict[str, Any] | None:
    """Read the newest FAILTEST record to demonstrate chunks and the outbox job survived failure."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT d.doc_id, d.file_name, d.subject, j.job_id, j.status
                FROM documents AS d
                LEFT JOIN ingestion_jobs AS j ON j.doc_id = d.doc_id
                WHERE d.file_name = %s AND d.subject = %s
                ORDER BY d.doc_id DESC
                LIMIT 1
                """,
                (file_name, FAILURE_SUBJECT),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (row[0],))
            chunk_count = cursor.fetchone()[0]
    finally:
        connection.close()

    return {
        "doc_id": row[0],
        "file_name": row[1],
        "subject": row[2],
        "job_id": row[3],
        "job_status": row[4],
        "chunk_count": chunk_count,
    }


def _print_failure_result(document: dict[str, Any] | None) -> None:
    """Print the exact durable state needed to verify no PostgreSQL data was lost."""
    if document is None:
        print("No FAILTEST document was found in PostgreSQL.")
        return

    print(f"Document row: doc_id={document['doc_id']}, file_name={document['file_name']}, subject={document['subject']}")
    print(f"Chunks saved: {document['chunk_count']}")
    print(f"Ingestion job: job_id={document['job_id']}, status={document['job_status']}")
    if document["chunk_count"] > 0 and document["job_status"] == "PENDING":
        print("PASS: PostgreSQL retained chunks and left the job PENDING while Qdrant was unavailable.")
    else:
        print("FAIL: expected saved chunks and a PENDING ingestion job.")


def cleanup_failtest() -> None:
    """Delete all FAILTEST rows and payload-filtered points so this manual consistency test is repeatable."""
    client = _qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="subject",
                        match=MatchValue(value=FAILURE_SUBJECT),
                    )
                ]
            )
        ),
        wait=True,
    )

    connection = _postgres_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM ingestion_jobs
                    WHERE doc_id IN (
                        SELECT doc_id FROM documents WHERE subject = %s
                    )
                    """,
                    (FAILURE_SUBJECT,),
                )
                jobs_deleted = cursor.rowcount
                cursor.execute("DELETE FROM documents WHERE subject = %s", (FAILURE_SUBJECT,))
                documents_deleted = cursor.rowcount
    finally:
        connection.close()

    print(
        f"Cleanup complete: deleted {documents_deleted} FAILTEST document(s), "
        f"{jobs_deleted} job(s), and matching Qdrant points."
    )


def run_failure_test(pdf_path: str | Path) -> None:
    """Pause for a manual Qdrant outage, then prove the failed vector write did not lose PostgreSQL data."""
    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf}")

    container_name = qdrant_container_name()
    print(f"Qdrant container: {container_name}")
    print(f"In another terminal, stop Qdrant with: docker stop {container_name}")
    input("After Qdrant has stopped, press Enter to run the FAILTEST ingestion: ")

    try:
        ingest_pdf(pdf, FAILURE_SUBJECT, user_id=1)
        print("Unexpected result: ingestion completed. Qdrant may still be running.")
    except Exception as error:
        print(f"Expected ingestion failure: {error}")

    _print_failure_result(_latest_failure_document(pdf.name))
    print(f"Start Qdrant again with: docker start {container_name}")


def main() -> None:
    """Run the manual failure proof for a PDF or remove all FAILTEST data with --cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", nargs="?", help="Small PDF to ingest with subject FAILTEST")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all FAILTEST documents, chunks, jobs, and Qdrant points",
    )
    args = parser.parse_args()

    if args.cleanup:
        cleanup_failtest()
        return
    if not args.pdf_path:
        parser.error("pdf_path is required unless --cleanup is used")
    run_failure_test(args.pdf_path)


if __name__ == "__main__":
    main()
