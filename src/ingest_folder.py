"""Ingest every PDF under data/<SUBJECT>/ using the single-PDF ingestion workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Running "py src/ingest_folder.py" sets sys.path to src/, not the project root.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import COLLECTION_NAME, ingest_pdf


load_dotenv(PROJECT_ROOT / ".env")


def _postgres_connection():
    """Open PostgreSQL through .env so duplicate checks use the durable source of truth."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _qdrant_client() -> QdrantClient:
    """Open Qdrant through .env to count the vector index after the batch finishes."""
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("Missing required environment variable: QDRANT_URL")
    return QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)


def document_exists(file_name: str, subject: str) -> bool:
    """Skip matching documents because duplicate rows and vectors would distort search and benchmarks."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM documents
                    WHERE file_name = %s AND subject = %s
                )
                """,
                (file_name, subject),
            )
            return cursor.fetchone()[0]
    finally:
        connection.close()


def _folder_pdfs(folder: Path) -> list[Path]:
    """Find PDFs recursively so every subject folder can be processed in one repeatable batch."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def _subject_for_pdf(folder: Path, pdf_path: Path) -> str:
    """Use the first folder below data as the subject, keeping document metadata consistent."""
    relative_path = pdf_path.relative_to(folder)
    if len(relative_path.parts) < 2:
        raise ValueError("expected layout: data/<SUBJECT>/<file>.pdf")
    return relative_path.parts[0]


def _totals() -> tuple[int, int]:
    """Count both stores to make the PostgreSQL-to-Qdrant ingestion result observable."""
    connection = _postgres_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            postgres_chunks = cursor.fetchone()[0]
    finally:
        connection.close()

    qdrant_points = _qdrant_client().count(
        collection_name=COLLECTION_NAME, exact=True
    ).count
    return postgres_chunks, qdrant_points


def _print_summary(rows: list[dict[str, Any]], totals: tuple[int, int]) -> None:
    """Print a small batch report so users can compare relational rows with vector points."""
    headers = ("file", "subject", "chunks", "job status")
    values = [
        (row["file"], row["subject"], str(row["chunks"]), row["job_status"])
        for row in rows
    ]
    widths = [
        max(len(header), *(len(value[index]) for value in values)) if values else len(header)
        for index, header in enumerate(headers)
    ]

    def format_row(values_to_print: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(width) for value, width in zip(values_to_print, widths, strict=True)
        )

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for value in values:
        print(format_row(value))
    print(f"Total chunks in PostgreSQL: {totals[0]}")
    print(f"Total points in Qdrant: {totals[1]}")


def ingest_folder(folder: str | Path = "data", user_id: int = 1) -> None:
    """Ingest a subject-organized folder while safely continuing after skipped or failed PDFs."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    rows: list[dict[str, Any]] = []
    for pdf_path in _folder_pdfs(folder_path):
        subject = "UNKNOWN"
        try:
            subject = _subject_for_pdf(folder_path, pdf_path)
            if document_exists(pdf_path.name, subject):
                print(f"skipped: {pdf_path.name}")
                rows.append(
                    {
                        "file": pdf_path.name,
                        "subject": subject,
                        "chunks": "-",
                        "job_status": "SKIPPED",
                    }
                )
                continue

            result = ingest_pdf(pdf_path, subject, user_id)
            rows.append(
                {
                    "file": pdf_path.name,
                    "subject": subject,
                    "chunks": result["chunk_count"],
                    "job_status": result["job_status"],
                }
            )
        except Exception as error:
            print(f"failed: {pdf_path.name} - {error}")
            rows.append(
                {
                    "file": pdf_path.name,
                    "subject": subject,
                    "chunks": "-",
                    "job_status": f"FAILED: {error}",
                }
            )

    _print_summary(rows, _totals())


def main() -> None:
    """Parse the folder and optional user ID for repeatable command-line batch ingestion."""
    parser = argparse.ArgumentParser(description="Ingest PDFs from data/<SUBJECT>/ folders.")
    parser.add_argument("folder", nargs="?", default="data", help="Root data folder")
    parser.add_argument("--user-id", type=int, default=1, help="PostgreSQL user ID")
    args = parser.parse_args()
    ingest_folder(args.folder, args.user_id)


if __name__ == "__main__":
    main()
