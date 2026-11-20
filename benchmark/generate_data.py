"""
Benchmark dataset generator for the Semantic Notes Search Engine.

Creates three benchmark datasets (1k, 10k, 50k chunks) in dedicated Postgres
tables and Qdrant collections.  Real vectors are reused from notes_chunks (no
re-embedding).  Padded copies add small Gaussian noise so HNSW is not filled
with exact duplicate vectors.  Results are saved to benchmark/results/dataset_build.csv.

Usage:
    py benchmark/generate_data.py                          # all three sizes
    py benchmark/generate_data.py --sizes 1000 10000       # subset
    py benchmark/generate_data.py --sizes 1000 --force     # overwrite existing
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
RESULTS_CSV = RESULTS_DIR / "dataset_build.csv"

# The real production collection we scroll vectors from.  Never touched.
SOURCE_COLLECTION = "notes_chunks"

# Fixed random seed so the datasets are identical on every rebuild.
# DBMS concept: deterministic test data lets you repeat experiments and compare
# results across runs without worrying about sampling variance.
RNG_SEED = 42

# Qdrant HNSW config to match the production collection.
VECTOR_SIZE = 384
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
UPLOAD_BATCH_SIZE = 256

# Lower threshold to trigger HNSW indexing on small collections.
# Qdrant's default indexing_threshold is about 20,000 KB, so small collections
# (like 1k and 10k) stay unindexed with indexed_vectors_count at 0.
# A low positive threshold (100 KB, well below the ~1,500 KB size of 1k vectors)
# ensures Qdrant builds the HNSW graph instead of leaving vectors unindexed
# (note: threshold=0 disables vector indexing in Qdrant).
HNSW_INDEXING_THRESHOLD_KB = 100

# Noise magnitude for padded vector perturbation.
# DBMS concept: tiny Gaussian noise prevents the HNSW graph from collapsing
# duplicate nodes into a degenerate structure, keeping benchmark measurements
# representative of real workloads.
NOISE_STD = 0.01

# Mapping from size label to table / collection names.  Hard-coded so the
# script never accidentally touches the production table or collection.
SIZE_CONFIG: dict[int, dict[str, str]] = {
    1_000: {"table": "bench_chunks_1k", "collection": "bench_1k"},
    10_000: {"table": "bench_chunks_10k", "collection": "bench_10k"},
    50_000: {"table": "bench_chunks_50k", "collection": "bench_50k"},
}

ALLOWED_TABLES = {v["table"] for v in SIZE_CONFIG.values()}
ALLOWED_COLLECTIONS = {v["collection"] for v in SIZE_CONFIG.values()}

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    """Read a required environment variable so credentials stay outside source control."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _pg_connect() -> psycopg2.extensions.connection:
    """Open a fresh PostgreSQL connection from .env for each major operation."""
    return psycopg2.connect(_require_env("POSTGRES_URL"))


def _qdrant_client() -> QdrantClient:
    """Create the Qdrant client from .env so the benchmark targets the same instance."""
    return QdrantClient(
        url=_require_env("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def _assert_safe_table(table: str) -> None:
    """
    Reject any table name that is not one of the three bench tables.

    DBMS concept: application-level allowlisting is a second layer of defence
    against SQL injection and accidental production writes when the script is
    modified or called programmatically.
    """
    if table not in ALLOWED_TABLES:
        raise ValueError(
            f"Refusing to operate on table '{table}'. "
            f"Only these are allowed: {ALLOWED_TABLES}"
        )


def _assert_safe_collection(collection: str) -> None:
    """Reject any Qdrant collection name that is not one of the three bench collections."""
    if collection not in ALLOWED_COLLECTIONS:
        raise ValueError(
            f"Refusing to operate on collection '{collection}'. "
            f"Only these are allowed: {ALLOWED_COLLECTIONS}"
        )


# ---------------------------------------------------------------------------
# Scroll real vectors from the production Qdrant collection
# ---------------------------------------------------------------------------

def scroll_real_chunks(client: QdrantClient) -> list[dict[str, Any]]:
    """
    Retrieve every point from notes_chunks, including its vector.

    DBMS concept: scrolling (cursor-based pagination) avoids loading the entire
    index into memory at once and matches how databases stream large result sets.
    We reuse existing vectors instead of re-embedding to save hours of CPU time
    and to keep the benchmark vectors byte-identical to production.
    """
    print(f"Scrolling all vectors from '{SOURCE_COLLECTION}' (this may take a moment)…")
    all_points: list[dict[str, Any]] = []
    offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=256,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in response:
            payload = point.payload or {}
            all_points.append(
                {
                    "chunk_id": str(point.id),
                    "vector": point.vector,          # list[float], length 384
                    "doc_id": payload.get("doc_id"),
                    "subject": payload.get("subject", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "page_no": payload.get("page_no"),
                    "chunk_text": payload.get("text", ""),
                }
            )
        if next_offset is None:
            break
        offset = next_offset

    print(f"  Scrolled {len(all_points):,} real chunks from '{SOURCE_COLLECTION}'.")
    return all_points


# ---------------------------------------------------------------------------
# Padded copy generation
# ---------------------------------------------------------------------------

def _add_noise_and_normalise(vector: list[float], rng: np.random.Generator) -> list[float]:
    """
    Add small Gaussian noise to a vector and re-normalise it to unit length.

    DBMS concept: re-normalisation keeps cosine-distance semantics intact (all
    production vectors are unit-normalised).  The tiny std=0.01 perturbation
    shifts the vector by ~1 %, which is enough to give HNSW distinct graph
    nodes while keeping semantic meaning nearly unchanged.
    """
    v = np.array(vector, dtype=np.float32)
    v += rng.normal(0.0, NOISE_STD, size=v.shape).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v.tolist()


def build_dataset(
    real_chunks: list[dict[str, Any]],
    target_size: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """
    Assemble a dataset of exactly `target_size` rows.

    Dataset design:
    - 1 k  → random sample of REAL chunks only (no padding needed).
    - 10 k → ALL real chunks + padded copies to reach 10,000.
    - 50 k → ALL real chunks + padded copies to reach 50,000.

    DBMS concept: keeping ALL real chunks in the larger datasets ensures the
    benchmark evaluates recall at scale rather than a biased subsample.  Padded
    rows must be labelled (is_padded=True) so the query runner in Step 6 can
    report precision over real data only.
    """
    n_real = len(real_chunks)

    if target_size <= n_real:
        # 1 k case: take a reproducible random sample of real chunks.
        indices = rng.choice(n_real, size=target_size, replace=False)
        selected = [real_chunks[i] for i in indices]
        rows = []
        for chunk in selected:
            rows.append(
                {
                    **chunk,
                    "is_padded": False,
                    "source_chunk_id": chunk["chunk_id"],
                }
            )
        return rows

    # 10 k / 50 k: include every real chunk first.
    rows: list[dict[str, Any]] = []
    for chunk in real_chunks:
        rows.append(
            {
                **chunk,
                "is_padded": False,
                "source_chunk_id": chunk["chunk_id"],
            }
        )

    # Then pad with noisy copies until we reach target_size.
    n_padded = target_size - n_real
    print(
        f"  Generating {n_padded:,} padded copies "
        f"(noise std={NOISE_STD}, seed={RNG_SEED})…"
    )
    # Cycle through real chunks so padded copies are spread evenly.
    for i in range(n_padded):
        source = real_chunks[i % n_real]
        rows.append(
            {
                "chunk_id": str(uuid4()),          # new UUID for the padded copy
                "vector": _add_noise_and_normalise(source["vector"], rng),
                "doc_id": source["doc_id"],
                "subject": source["subject"],
                "chunk_index": source["chunk_index"],
                "page_no": source["page_no"],
                "chunk_text": source["chunk_text"],
                "is_padded": True,
                "source_chunk_id": source["chunk_id"],
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Existence / count checks
# ---------------------------------------------------------------------------

def _pg_table_exists(table: str) -> bool:
    """Check whether a bench table already exists in the public schema."""
    conn = _pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = %s",
                (table,),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def _pg_row_count(table: str) -> int:
    """Return the exact row count for a bench table."""
    _assert_safe_table(table)
    conn = _pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 – table allowlisted
            return cur.fetchone()[0]
    finally:
        conn.close()


def _qdrant_point_count(client: QdrantClient, collection: str) -> int:
    """Return the number of indexed points in a Qdrant collection."""
    info = client.get_collection(collection)
    return info.points_count or 0


def _already_built(
    client: QdrantClient, table: str, collection: str, target_size: int
) -> bool:
    """
    Return True when both the table and collection exist with the correct count.

    DBMS concept: idempotent setup scripts are essential in reproducible research
    pipelines; re-running must not corrupt existing data.  Counts serve as a
    lightweight integrity check before skipping expensive builds.
    """
    if not _pg_table_exists(table):
        return False
    if not client.collection_exists(collection):
        return False
    pg_count = _pg_row_count(table)
    qd_count = _qdrant_point_count(client, collection)
    return pg_count == target_size and qd_count == target_size


# ---------------------------------------------------------------------------
# PostgreSQL table creation and loading
# ---------------------------------------------------------------------------

def _create_bench_table(table: str) -> None:
    """
    Create a bench table with the same columns as chunks plus is_padded and
    source_chunk_id.  doc_id and subject are plain columns (no foreign key)
    so padded rows with the same doc_id do not require matching documents rows.

    DBMS concept: dropping the foreign key on bench tables is intentional.
    A foreign key would force every padded doc_id to exist in the documents
    table, which would pollute the production schema.  Bench tables are
    self-contained scratch spaces.
    """
    _assert_safe_table(table)
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE {table} (
                        chunk_id        UUID        PRIMARY KEY,
                        doc_id          INTEGER     NOT NULL,
                        chunk_index     INTEGER     NOT NULL,
                        chunk_text      TEXT        NOT NULL,
                        page_no         INTEGER,
                        subject         TEXT        NOT NULL DEFAULT '',
                        is_padded       BOOLEAN     NOT NULL,
                        source_chunk_id UUID        NOT NULL
                    )
                    """  # noqa: S608 – table name allowlisted above
                )
        print(f"  Created Postgres table '{table}'.")
    finally:
        conn.close()


def _build_gin_index(table: str) -> float:
    """
    Build a GIN index on to_tsvector('english', chunk_text), matching the
    production index on the real chunks table.

    DBMS concept: each bench table gets its own GIN index because index build
    time is part of the benchmark.  A shared index on a view would hide the
    cost of indexing at scale.  GIN (Generalised Inverted Index) stores a
    posting list per lexeme, enabling O(log n) full-text lookup.

    Returns the build time in seconds.
    """
    _assert_safe_table(table)
    index_name = f"idx_{table}_fts"
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                t0 = time.perf_counter()
                cur.execute(
                    f"""
                    CREATE INDEX {index_name}
                    ON {table}
                    USING GIN (to_tsvector('english', chunk_text))
                    """  # noqa: S608 – table name allowlisted above
                )
                elapsed = time.perf_counter() - t0
        print(f"  GIN index '{index_name}' built in {elapsed:.2f}s.")
        return elapsed
    finally:
        conn.close()


def load_postgres(table: str, rows: list[dict[str, Any]]) -> tuple[float, float]:
    """
    Bulk-insert all rows into a bench table using execute_values, then build the GIN index.

    DBMS concept: execute_values sends a single multi-row INSERT instead of one
    statement per row, reducing round-trips and enabling PostgreSQL's bulk-insert
    path.  The GIN index is built after the data load because building it
    incrementally during inserts is slower than a one-shot sort-based build.

    Returns (pg_load_s, gin_build_s).
    """
    _assert_safe_table(table)
    conn = _pg_connect()
    try:
        t0 = time.perf_counter()
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {table}
                        (chunk_id, doc_id, chunk_index, chunk_text,
                         page_no, subject, is_padded, source_chunk_id)
                    VALUES %s
                    """,  # noqa: S608 – table name allowlisted above
                    [
                        (
                            row["chunk_id"],
                            row["doc_id"],
                            row["chunk_index"],
                            row["chunk_text"],
                            row["page_no"],
                            row["subject"],
                            row["is_padded"],
                            row["source_chunk_id"],
                        )
                        for row in rows
                    ],
                    page_size=1000,
                )
        pg_load_s = time.perf_counter() - t0
        print(
            f"  Loaded {len(rows):,} rows into '{table}' in {pg_load_s:.2f}s."
        )
    finally:
        conn.close()

    gin_build_s = _build_gin_index(table)
    return pg_load_s, gin_build_s


# ---------------------------------------------------------------------------
# Qdrant collection creation and loading
# ---------------------------------------------------------------------------

def _create_bench_collection(client: QdrantClient, collection: str) -> None:
    """
    Create a Qdrant collection with the same HNSW config as notes_chunks.

    DBMS concept: using identical HNSW parameters (m=16, ef_construct=100)
    ensures the benchmark measures the effect of dataset size, not index
    configuration differences.  m controls graph connectivity (memory vs.
    recall), ef_construct controls build-time search depth (quality vs. speed).
    The default indexing_threshold is about 20,000 KB, which leaves smaller
    collections (1k, 10k) unindexed with indexed_vectors_count at 0. A low
    positive threshold (100 KB) makes Qdrant build the HNSW graph so the polling
    loop gets a meaningful indexed count (0 would disable indexing entirely).
    """
    _assert_safe_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=HNSW_INDEXING_THRESHOLD_KB,
        ),
    )
    print(f"  Created Qdrant collection '{collection}'.")


def _wait_for_index_ready(
    client: QdrantClient, collection: str, expected: int, timeout_s: float = 180.0
) -> tuple[float, bool]:
    """
    Poll the collection status until the HNSW index is green and
    indexed_vectors_count equals points_count (and both equal expected).
    Gives up after timeout_s seconds and prints a clear diagnostic.

    DBMS concept: Qdrant builds the HNSW graph asynchronously after upload.
    Polling is necessary to measure the true index-ready latency, which is what
    Step 6 must wait for before running HNSW queries.  Without waiting, the
    first queries hit the flat (brute-force) fallback path.

    Returns (wait_s, index_complete).  index_complete is False when the timeout
    fires, so dataset_build.csv can flag incomplete index builds for diagnosis.
    """
    _assert_safe_collection(collection)
    print(f"  Waiting for HNSW index to be ready in '{collection}'…", end="", flush=True)
    t0 = time.perf_counter()
    poll_interval = 2.0  # seconds between status checks

    while True:
        info = client.get_collection(collection)
        status = str(info.status).lower()
        indexed = info.indexed_vectors_count or 0
        points  = info.points_count or 0
        if "green" in status and indexed == points == expected:
            elapsed = time.perf_counter() - t0
            print(f" ready in {elapsed:.2f}s (indexed={indexed:,}).")
            return elapsed, True
        elapsed = time.perf_counter() - t0
        if elapsed >= timeout_s:
            print(
                f"\n  WARNING: index not ready after {timeout_s:.0f}s "
                f"(status={status}, indexed={indexed:,}, points={points:,}, "
                f"expected={expected:,}). Continuing without full HNSW index."
            )
            return elapsed, False
        print(".", end="", flush=True)
        time.sleep(poll_interval)


def upload_to_qdrant(
    client: QdrantClient, collection: str, rows: list[dict[str, Any]]
) -> tuple[float, float, bool]:
    """
    Upload all rows as Qdrant points in batches of UPLOAD_BATCH_SIZE=256,
    then wait for the HNSW index to be fully built.

    DBMS concept: batching reduces gRPC round-trips and allows Qdrant's write
    path to coalesce segment flushes.  Payload fields stored here (doc_id,
    subject, chunk_index, page_no, is_padded) mirror the production collection
    so Step 6 can filter by subject or is_padded without schema changes.

    Returns (qdrant_upload_s, qdrant_index_ready_s, index_complete).
    """
    _assert_safe_collection(collection)
    t0 = time.perf_counter()
    total = len(rows)
    for start in range(0, total, UPLOAD_BATCH_SIZE):
        batch = rows[start : start + UPLOAD_BATCH_SIZE]
        points = [
            PointStruct(
                id=row["chunk_id"],
                vector=row["vector"],
                payload={
                    "doc_id": row["doc_id"],
                    "subject": row["subject"],
                    "chunk_index": row["chunk_index"],
                    "page_no": row["page_no"],
                    "is_padded": row["is_padded"],
                    "source_chunk_id": row["source_chunk_id"],
                },
            )
            for row in batch
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        pct = min(100, int((start + len(batch)) / total * 100))
        print(f"\r  Uploading to '{collection}': {pct}%", end="", flush=True)
    print()  # newline after progress

    upload_s = time.perf_counter() - t0
    print(f"  Uploaded {total:,} points in {upload_s:.2f}s.")

    index_ready_s, index_complete = _wait_for_index_ready(client, collection, total)
    return upload_s, index_ready_s, index_complete


# ---------------------------------------------------------------------------
# Drop helpers (require --force and an interactive YES prompt)
# ---------------------------------------------------------------------------

def _drop_pg_table(table: str) -> None:
    """
    Drop a bench table after an explicit YES confirmation.

    DBMS concept: DROP TABLE is irreversible; requiring --force plus a typed
    confirmation prevents accidental data loss in automated pipelines.
    """
    _assert_safe_table(table)
    answer = input(
        f"  Type YES to drop Postgres table '{table}': "
    ).strip()
    if answer != "YES":
        print("  Aborted.")
        return
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table}")  # noqa: S608
        print(f"  Dropped table '{table}'.")
    finally:
        conn.close()


def _drop_qdrant_collection(client: QdrantClient, collection: str) -> None:
    """Drop a bench Qdrant collection after an explicit YES confirmation."""
    _assert_safe_collection(collection)
    answer = input(
        f"  Type YES to drop Qdrant collection '{collection}': "
    ).strip()
    if answer != "YES":
        print("  Aborted.")
        return
    client.delete_collection(collection)
    print(f"  Dropped collection '{collection}'.")


# ---------------------------------------------------------------------------
# Subject count reporter
# ---------------------------------------------------------------------------

def _print_subject_counts(rows: list[dict[str, Any]], size_label: str) -> None:
    """
    Print per-subject chunk counts (real and padded) so the dataset balance is visible.

    DBMS concept: data distributions matter for benchmark validity.  An index
    scan that only ever hits one subject partition is not representative.
    """
    from collections import Counter
    real_subjects: Counter = Counter()
    padded_subjects: Counter = Counter()
    for row in rows:
        subj = row.get("subject") or "unknown"
        if row["is_padded"]:
            padded_subjects[subj] += 1
        else:
            real_subjects[subj] += 1

    all_subjects = sorted(set(real_subjects) | set(padded_subjects))
    print(f"  Subject breakdown for {size_label}:")
    for subj in all_subjects:
        r = real_subjects.get(subj, 0)
        p = padded_subjects.get(subj, 0)
        print(f"    {subj}: {r:,} real, {p:,} padded")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _save_results(records: list[dict[str, Any]]) -> None:
    """
    Append (or create) benchmark/results/dataset_build.csv with one row per size.

    DBMS concept: persisting build metrics lets future runs compare load
    performance without re-running the full setup, mirroring how database
    systems keep pg_stat_user_tables and similar system views.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "size",
        "real_chunks",
        "padded_chunks",
        "pg_load_s",
        "gin_build_s",
        "qdrant_upload_s",
        "qdrant_index_ready_s",
        "index_complete",
    ]
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record)
    print(f"\nResults saved to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Main build routine for one size
# ---------------------------------------------------------------------------

def build_one_size(
    size: int,
    real_chunks: list[dict[str, Any]],
    client: QdrantClient,
    rng: np.random.Generator,
    force: bool,
) -> dict[str, Any]:
    """
    Build the Postgres table and Qdrant collection for one dataset size.

    DBMS concept: giving each size its own table and collection isolates I/O
    patterns so a scan of 1,000 rows does not share buffer-pool pages with a
    50,000-row scan.  Shared tables would skew cache-hit ratios and invalidate
    the latency comparison.

    Returns a metrics dict suitable for dataset_build.csv.
    """
    cfg = SIZE_CONFIG[size]
    table = cfg["table"]
    collection = cfg["collection"]
    size_label = f"{size:,}"

    print(f"\n{'='*60}")
    print(f"Building dataset: {size_label} chunks  (table={table}, collection={collection})")
    print(f"{'='*60}")

    # --- Check if already built ---
    if not force and _already_built(client, table, collection, size):
        print(
            f"  Skipping {size_label}: table and collection already exist "
            f"with the correct counts. Use --force to rebuild."
        )
        return {}  # caller skips CSV row

    # --- Drop existing artefacts when --force is used ---
    if force:
        if _pg_table_exists(table):
            _drop_pg_table(table)
        if client.collection_exists(collection):
            _drop_qdrant_collection(client, collection)
        # Re-check after potential abort
        if _pg_table_exists(table) or client.collection_exists(collection):
            print(f"  Skipping {size_label}: drop was aborted.")
            return {}

    # --- Assemble rows ---
    print(f"  Assembling {size_label} rows (seed={RNG_SEED})…")
    rows = build_dataset(real_chunks, size, rng)
    n_real = sum(1 for r in rows if not r["is_padded"])
    n_padded = sum(1 for r in rows if r["is_padded"])
    print(f"  Rows: {n_real:,} real + {n_padded:,} padded = {len(rows):,} total")
    _print_subject_counts(rows, size_label)

    # --- Postgres ---
    _create_bench_table(table)
    pg_load_s, gin_build_s = load_postgres(table, rows)

    # --- Qdrant ---
    _create_bench_collection(client, collection)
    qdrant_upload_s, qdrant_index_ready_s, index_complete = upload_to_qdrant(client, collection, rows)

    # --- Final parity check ---
    pg_count = _pg_row_count(table)
    qd_count = _qdrant_point_count(client, collection)
    if pg_count == qd_count == size:
        print(
            f"  ✓ Parity check passed: "
            f"Postgres={pg_count:,}, Qdrant={qd_count:,}"
        )
    else:
        print(
            f"  ✗ PARITY MISMATCH: "
            f"expected {size}, Postgres={pg_count:,}, Qdrant={qd_count:,}"
        )

    return {
        "size": size,
        "real_chunks": n_real,
        "padded_chunks": n_padded,
        "pg_load_s": round(pg_load_s, 3),
        "gin_build_s": round(gin_build_s, 3),
        "qdrant_upload_s": round(qdrant_upload_s, 3),
        "qdrant_index_ready_s": round(qdrant_index_ready_s, 3),
        "index_complete": index_complete,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments so the caller can target specific sizes or force a rebuild."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=sorted(SIZE_CONFIG.keys()),
        choices=sorted(SIZE_CONFIG.keys()),
        metavar="SIZE",
        help=(
            "Dataset sizes to build. "
            f"Allowed: {sorted(SIZE_CONFIG.keys())}. "
            "Default: all three."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Drop and rebuild sizes whose table/collection already exist. "
            "Each drop requires typing YES at the prompt."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point: scroll real vectors once, then build each requested size.

    DBMS concept: scrolling once and reusing the in-memory list avoids repeated
    network round-trips to Qdrant.  The list is never mutated; each size build
    calls build_dataset() which returns a fresh copy, so builds are independent.
    """
    args = _parse_args()
    sizes = sorted(set(args.sizes))

    # Validate that no production artefacts are targeted.
    for size in sizes:
        cfg = SIZE_CONFIG[size]
        _assert_safe_table(cfg["table"])
        _assert_safe_collection(cfg["collection"])

    client = _qdrant_client()

    # Scroll real vectors from production (read-only, no writes to notes_chunks).
    real_chunks = scroll_real_chunks(client)
    if not real_chunks:
        sys.exit("ERROR: No chunks found in notes_chunks. Ingest some PDFs first.")

    # Fixed RNG – reset before each size so each build is independent.
    # (The rng is passed into build_dataset to keep all randomness traceable.)
    csv_records: list[dict[str, Any]] = []

    for size in sizes:
        # Reset RNG per size so smaller sizes are sub-samples of larger ones.
        rng = np.random.default_rng(RNG_SEED)
        record = build_one_size(size, real_chunks, client, rng, force=args.force)
        if record:
            csv_records.append(record)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Size':>8}  {'Real':>8}  {'Padded':>8}  {'PG rows':>8}  {'QD pts':>8}  {'Match':>6}")
    for size in sizes:
        cfg = SIZE_CONFIG[size]
        if _pg_table_exists(cfg["table"]) and client.collection_exists(cfg["collection"]):
            pg = _pg_row_count(cfg["table"])
            qd = _qdrant_point_count(client, cfg["collection"])
            match = "✓" if pg == qd == size else "✗"
            print(
                f"{size:>8,}  {pg - (size - min(size, len(real_chunks))):>8,}  "
                f"{max(0, size - len(real_chunks)):>8,}  "
                f"{pg:>8,}  {qd:>8,}  {match:>6}"
            )

    if csv_records:
        _save_results(csv_records)

    print("\nDone.")


if __name__ == "__main__":
    main()
