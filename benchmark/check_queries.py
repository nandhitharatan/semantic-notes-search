"""
Diagnostic script: check how many rows match each query keyword in every bench table.

For each of the 20 entries in queries.json and each dataset size (1k, 10k, 50k),
prints:
  - ILIKE match count  (WHERE chunk_text ILIKE '%keyword%')
  - FTS match count    (WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery(...))

Marks any zero match count as "ZERO" and prints a final list of keywords that
have zero matches at ANY size.  No model loading.  Read-only against bench tables.

Usage:
    py benchmark/check_queries.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
QUERIES_FILE  = BENCHMARK_DIR / "queries.json"

# Hard-coded allowlist — this script never touches production tables.
ALLOWED_TABLES = {"bench_chunks_1k", "bench_chunks_10k", "bench_chunks_50k"}

SIZE_CONFIG: dict[int, str] = {
    1_000:  "bench_chunks_1k",
    10_000: "bench_chunks_10k",
    50_000: "bench_chunks_50k",
}

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _pg_connect() -> psycopg2.extensions.connection:
    """Open a PostgreSQL connection from .env."""
    url = os.getenv("POSTGRES_URL")
    if not url:
        raise RuntimeError("Missing required environment variable: POSTGRES_URL")
    return psycopg2.connect(url)


def _assert_safe_table(table: str) -> None:
    """Reject any table not in the bench allowlist."""
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Refusing to query '{table}'. Allowed: {sorted(ALLOWED_TABLES)}")


def count_like(conn: psycopg2.extensions.connection, table: str, keyword: str) -> int:
    """Return COUNT(*) for ILIKE '%keyword%' — no LIMIT, full table scan."""
    _assert_safe_table(table)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {table} WHERE chunk_text ILIKE %s",  # noqa: S608
            (f"%{keyword}%",),
        )
        return cur.fetchone()[0]


def count_fts(conn: psycopg2.extensions.connection, table: str, keyword: str) -> int:
    """Return COUNT(*) for the GIN full-text match — no LIMIT."""
    _assert_safe_table(table)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) FROM {table}
            WHERE to_tsvector('english', chunk_text)
                  @@ plainto_tsquery('english', %s)
            """,  # noqa: S608
            (keyword,),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(n: int) -> str:
    """Format count; mark zero prominently."""
    return "ZERO" if n == 0 else str(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(QUERIES_FILE, encoding="utf-8") as fh:
        queries = json.load(fh)

    sizes = sorted(SIZE_CONFIG.keys())
    tables = [SIZE_CONFIG[s] for s in sizes]
    size_labels = [f"{s:,}" for s in sizes]

    try:
        conn = _pg_connect()
    except Exception as exc:
        print(f"ERROR: cannot connect to PostgreSQL: {exc}")
        return

    # Collect results: (qid, keyword, {size: (like_n, fts_n)})
    results: list[tuple[str, str, dict[int, tuple[int, int]]]] = []

    for q in queries:
        qid     = q["id"]
        keyword = q["keyword"]
        counts: dict[int, tuple[int, int]] = {}
        for size in sizes:
            table = SIZE_CONFIG[size]
            try:
                like_n = count_like(conn, table, keyword)
                fts_n  = count_fts(conn, table, keyword)
            except Exception as exc:
                print(f"  ERROR [{qid} @ {size:,}]: {exc}")
                like_n = -1
                fts_n  = -1
            counts[size] = (like_n, fts_n)
        results.append((qid, keyword, counts))

    conn.close()

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    col_w   = 10   # width per count cell
    kw_w    = 24   # width for keyword column
    id_w    = 10   # width for id column
    method_w = 6   # "LIKE" / "FTS"

    # Header rows
    size_header = "".join(f"{lbl:^{col_w * 2}}" for lbl in size_labels)
    sub_header  = "".join(("LIKE".center(col_w) + "FTS".center(col_w)) for _ in sizes)
    sep_line    = "-" * (id_w + 1 + kw_w + len(sub_header))

    print(f"\n{'ID':<{id_w}} {'Keyword':<{kw_w}}{size_header}")
    print(f"{'':<{id_w}} {'':<{kw_w}}{sub_header}")
    print(sep_line)

    zero_keywords: list[str] = []  # keywords with at least one ZERO

    for qid, keyword, counts in results:
        has_zero = False
        row = f"{qid:<{id_w}} {keyword:<{kw_w}}"
        for size in sizes:
            like_n, fts_n = counts[size]
            lf = _fmt(like_n)
            ff = _fmt(fts_n)
            if lf == "ZERO" or ff == "ZERO":
                has_zero = True
            row += f"{lf:^{col_w}}{ff:^{col_w}}"
        print(row)
        if has_zero:
            zero_keywords.append(keyword)

    print(sep_line)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\nTotal queries checked: {len(results)}")
    if zero_keywords:
        print(f"\nKeywords with ZERO matches at any size ({len(zero_keywords)}):")
        for kw in zero_keywords:
            print(f"  - {kw}")
    else:
        print("\nAll keywords return at least one match at every size. ✓")


if __name__ == "__main__":
    main()
