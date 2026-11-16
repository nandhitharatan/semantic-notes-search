"""Compare the three search methods for one question without changing stored data."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Running this file directly makes benchmark/ the first import path.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search_sql import search_fts, search_like
from src.search_vector import search_vector


load_dotenv(PROJECT_ROOT / ".env")


def _postgres_connection():
    """Open PostgreSQL from .env so vector results can be shown with their source pages."""
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(postgres_url)


def _vector_pages(results: list[dict[str, Any]]) -> list[Any]:
    """Look up page provenance for vector hits because the fixed vector API returns doc_id and text."""
    connection = _postgres_connection()
    try:
        pages: list[Any] = []
        with connection.cursor() as cursor:
            for result in results:
                cursor.execute(
                    """
                    SELECT page_no
                    FROM chunks
                    WHERE doc_id = %s AND chunk_text = %s
                    LIMIT 1
                    """,
                    (result.get("doc_id"), result.get("text", "")),
                )
                row = cursor.fetchone()
                pages.append(row[0] if row else "?")
        return pages
    finally:
        connection.close()


def _run_search(
    label: str,
    function: Callable[..., list[dict[str, Any]]],
    question: str,
    subject: str | None = None,
) -> tuple[str, list[dict[str, Any]], float]:
    """Time one search call so the demo exposes the latency trade-off between indexes."""
    started = time.perf_counter()
    if label == "Vector Search":
        results = function(question, 3, subject=subject)
    else:
        results = function(question, 3)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return label, results, elapsed_ms


def _print_results(
    label: str,
    results: list[dict[str, Any]],
    elapsed_ms: float,
    pages: list[Any] | None = None,
) -> None:
    """Print counts, latency, and short page-tagged snippets for one search method."""
    print(f"{label}: {len(results)} results, {elapsed_ms:.2f} ms")
    for index, result in enumerate(results):
        page = pages[index] if pages is not None else result.get("page_no", "?")
        text = result.get("chunk_text", result.get("text", ""))
        print(f"  page {page}: {text[:120].replace(chr(10), ' ')}")


def main() -> None:
    """Run LIKE, FTS, and HNSW searches for the command-line question."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question or search phrase")
    parser.add_argument("--subject", default=None, help="Optional subject filter for vector search")
    args = parser.parse_args()

    like_label, like_results, like_ms = _run_search("SQL LIKE", search_like, args.question)
    fts_label, fts_results, fts_ms = _run_search("SQL Full-Text", search_fts, args.question)
    vector_label, vector_results, vector_ms = _run_search(
        "Vector Search", search_vector, args.question, args.subject
    )

    _print_results(like_label, like_results, like_ms)
    _print_results(fts_label, fts_results, fts_ms)
    _print_results(vector_label, vector_results, vector_ms, _vector_pages(vector_results))


if __name__ == "__main__":
    main()
