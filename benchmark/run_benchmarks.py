"""
Benchmark runner for the Semantic Notes Search Engine.

Measures latency and recall for five search methods across three dataset sizes:

  1. sql_like          – PostgreSQL ILIKE substring scan (sequential, no index)
  2. sql_fts           – PostgreSQL GIN full-text search with ts_rank ordering
  3. qdrant_hnsw_ef16  – Qdrant HNSW approximate search, query-time hnsw_ef=16
  4. qdrant_hnsw_ef128 – Qdrant HNSW approximate search, query-time hnsw_ef=128
  5. qdrant_exact      – Qdrant exhaustive (brute-force) search; ground truth for recall

WHY HNSW IS APPROXIMATE
------------------------
HNSW (Hierarchical Navigable Small World) builds a layered proximity graph over
the vector space.  At query time it traverses the graph greedily, exploring
`hnsw_ef` candidate neighbours before returning the top-k.  A larger ef value
visits more candidates, improving recall at the cost of higher latency.  Because
the traversal is greedy it can miss true nearest neighbours, making HNSW
approximate.  ef=128 is a generous setting that typically achieves >0.99 recall;
ef=16 is a fast, low-recall setting that illustrates the trade-off.

WHY EXACT SEARCH IS GROUND TRUTH
----------------------------------
SearchParams(exact=True) tells Qdrant to skip the HNSW graph and perform a full
linear scan over all vectors (dot-product / cosine distance).  This is O(n) but
guaranteed to return the true nearest neighbours.  We use it as the ground truth
against which HNSW recall is measured.

WHY FILTERED SEARCHES ARE EXCLUDED
------------------------------------
Qdrant's full_scan_threshold is set to 10,000 (vectors), which means a filtered
query on a small payload (e.g., subject='DBMS') can bypass the HNSW index
entirely and fall back to a filtered linear scan.  Including such queries would
make it impossible to compare HNSW and exact methods fairly, so ALL benchmark
queries run without any payload filter.

RECALL DEFINITIONS
-------------------
  recall_at_5_by_point  – strict: fraction of exact top-5 point UUIDs also
                          returned by the HNSW run, averaged over 20 queries.
  recall_at_5_by_source – lenient: fraction of exact top-5 SOURCE chunk UUIDs
                          also returned (after resolving padded copies back to
                          their real source via the source_chunk_id payload).
  Lenient recall is higher because padded copies are near-duplicates of the same
  text; if HNSW returns a padded copy of the true nearest neighbour it is still
  semantically correct.  Both columns are reported so the difference is visible.

TIMING RULES
-------------
  - SentenceTransformer model is loaded ONCE before any timed section.
  - All 20 questions are embedded in a single batch; embedding time is recorded
    per query but NOT included in Qdrant latency.
  - Per (query, method): 2 warm-up runs (not recorded), then N_RUNS recorded
    runs with time.perf_counter().  --quick overrides to 2 runs.
  - Reported statistics (p50, p95, mean) are computed over all recorded runs
    across all queries for a given (size, method).
  - SQL latency = cursor.execute + fetchall only.
  - Qdrant latency = query_points call only (precomputed vector).
  - A fresh DB connection and Qdrant client are opened per size.

Usage:
    py benchmark/run_benchmarks.py                              # all sizes, 10 runs
    py benchmark/run_benchmarks.py --sizes 1000 10000           # subset
    py benchmark/run_benchmarks.py --quick                      # 3 queries, 2 runs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
QUERIES_FILE = BENCHMARK_DIR / "queries.json"

# Embedding model — must match the model used to build the bench collections.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Hard-coded allowlist: the benchmark only reads from these tables/collections.
# Any name not in these sets is rejected before a query is issued.
ALLOWED_TABLES = {"bench_chunks_1k", "bench_chunks_10k", "bench_chunks_50k"}
ALLOWED_COLLECTIONS = {"bench_1k", "bench_10k", "bench_50k"}

SIZE_CONFIG: dict[int, dict[str, str]] = {
    1_000:  {"table": "bench_chunks_1k",  "collection": "bench_1k"},
    10_000: {"table": "bench_chunks_10k", "collection": "bench_10k"},
    50_000: {"table": "bench_chunks_50k", "collection": "bench_50k"},
}

# Default benchmark parameters.
N_WARMUP = 2    # warm-up runs per (query, method) — not recorded
N_RUNS   = 10   # recorded runs per (query, method)
LIMIT    = 5    # top-k for every method

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    """Read a required environment variable; raise clearly if absent."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _pg_connect() -> psycopg2.extensions.connection:
    """Open a fresh PostgreSQL connection from .env credentials."""
    return psycopg2.connect(_require_env("POSTGRES_URL"))


def _qdrant_client() -> QdrantClient:
    """Create a Qdrant client from .env — same instance used throughout the project."""
    return QdrantClient(
        url=_require_env("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


# ---------------------------------------------------------------------------
# Safety guards — read-only allowlisting
# ---------------------------------------------------------------------------

def _assert_safe_table(table: str) -> None:
    """
    Reject any table name not in the bench allowlist.

    DBMS concept: application-level allowlisting is a second layer of defence
    against typos or programmatic errors that could accidentally read (or, in a
    write path, corrupt) production data.  This script is read-only, but the
    guard is kept for defence-in-depth and to mirror generate_data.py's pattern.
    """
    if table not in ALLOWED_TABLES:
        raise ValueError(
            f"Refusing to query table '{table}'. "
            f"Only these are allowed: {sorted(ALLOWED_TABLES)}"
        )


def _assert_safe_collection(collection: str) -> None:
    """Reject any Qdrant collection name not in the bench allowlist."""
    if collection not in ALLOWED_COLLECTIONS:
        raise ValueError(
            f"Refusing to query collection '{collection}'. "
            f"Only these are allowed: {sorted(ALLOWED_COLLECTIONS)}"
        )


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(path: Path, quick: bool) -> list[dict[str, str]]:
    """
    Load benchmark queries from JSON.  --quick caps the list at 3 entries.

    Each entry must have: id, subject, keyword, question.
    """
    with open(path, encoding="utf-8") as fh:
        queries = json.load(fh)
    if quick:
        queries = queries[:3]
    return queries


# ---------------------------------------------------------------------------
# Embedding (done once, before any timed section)
# ---------------------------------------------------------------------------

def embed_queries(
    queries: list[dict[str, str]],
) -> tuple[list[list[float]], dict[str, float]]:
    """
    Load the SentenceTransformer model once and embed every question in a
    single batch pass.  Returns (vectors, embedding_times_by_id).

    embedding_times_by_id maps query id -> ms spent on that query's share
    of the batch.  We attribute the total batch time proportionally across
    queries; this is an approximation but reasonable because transformer
    inference time scales roughly linearly with token count at the same
    batch size.

    WHY ONE BATCH: batching feeds the GPU/CPU matrix kernel once instead of
    N times, dramatically reducing per-query overhead from Python dispatch and
    tokeniser calls.  The embedding time is recorded separately and excluded
    from Qdrant latency so the two concerns are measured independently.
    """
    # Import here so the model load is deferred until this function is called.
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'…")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    questions = [q["question"] for q in queries]

    t0 = time.perf_counter()
    vectors_np = model.encode(
        questions,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=len(questions),  # one shot
    )
    total_embed_ms = (time.perf_counter() - t0) * 1000.0

    # Attribute time proportionally across queries.
    per_query_ms = total_embed_ms / len(queries)
    embedding_times = {q["id"]: per_query_ms for q in queries}

    vectors = [v.tolist() for v in vectors_np]
    print(
        f"Embedded {len(queries)} question(s) in {total_embed_ms:.1f} ms "
        f"({per_query_ms:.1f} ms/query)."
    )
    return vectors, embedding_times


# ---------------------------------------------------------------------------
# SQL search methods
# ---------------------------------------------------------------------------

def run_sql_like(
    conn: psycopg2.extensions.connection,
    table: str,
    keyword: str,
    limit: int,
) -> list[str]:
    """
    ILIKE substring scan: WHERE chunk_text ILIKE '%keyword%' ORDER BY chunk_id LIMIT k.

    DBMS concept: ILIKE is a sequential scan (or index scan if a pg_trgm index
    exists, but we have none) that checks every row for the pattern.  Latency
    grows linearly with table size, making it the slowest method at 50 k but a
    useful baseline — it has no vocabulary limitation and matches any substring.

    Returns a list of chunk_id strings (top-k, deterministic by chunk_id order).
    """
    _assert_safe_table(table)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT chunk_id
            FROM {table}
            WHERE chunk_text ILIKE %s
            ORDER BY chunk_id
            LIMIT %s
            """,  # noqa: S608 — table name allowlisted above
            (f"%{keyword}%", limit),
        )
        rows = cur.fetchall()
    return [str(r[0]) for r in rows]


def run_sql_fts(
    conn: psycopg2.extensions.connection,
    table: str,
    keyword: str,
    limit: int,
) -> list[str]:
    """
    GIN full-text search: to_tsvector @@ plainto_tsquery, ordered by ts_rank.

    DBMS concept: PostgreSQL's GIN (Generalised Inverted Index) stores a
    posting list per lexeme (stemmed word).  A tsvector query hits the index
    for O(log n + matches) lookup instead of a full table scan.  ts_rank
    scores documents by term frequency, putting the most relevant chunks first.
    This method ignores stop words and applies English stemming, so 'indexing'
    matches 'index' but 'GIN' would not match 'gin' after case folding.

    Returns a list of chunk_id strings (top-k, by ts_rank descending then chunk_id).
    """
    _assert_safe_table(table)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT chunk_id
            FROM {table}
            WHERE to_tsvector('english', chunk_text)
                  @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(
                         to_tsvector('english', chunk_text),
                         plainto_tsquery('english', %s)
                     ) DESC,
                     chunk_id
            LIMIT %s
            """,  # noqa: S608 — table name allowlisted above
            (keyword, keyword, limit),
        )
        rows = cur.fetchall()
    return [str(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Qdrant search methods
# ---------------------------------------------------------------------------

def run_qdrant(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int,
    hnsw_ef: int | None,
    exact: bool,
) -> list[str]:
    """
    Issue a Qdrant query_points call with the given SearchParams.

    No payload filter is applied.  See the module docstring for why filtered
    queries are excluded from this benchmark.

    Returns a list of point id strings (top-k, by score descending).
    """
    _assert_safe_collection(collection)
    response = client.query_points(
        collection_name=collection,
        query=vector,
        search_params=SearchParams(hnsw_ef=hnsw_ef, exact=exact),
        limit=limit,
        with_payload=True,  # needed to read source_chunk_id for lenient recall
    )
    return response.points


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def _timed_sql(
    conn: psycopg2.extensions.connection,
    table: str,
    fn,           # callable(conn, table, kw, limit) -> list[str]
    keyword: str,
    limit: int,
    n_warmup: int,
    n_runs: int,
) -> tuple[list[str], list[float]]:
    """
    Run fn n_warmup+n_runs times, discarding the first n_warmup results.

    Times only cursor.execute + fetchall (the fn body); connection overhead is
    excluded.  Returns (last_result, recorded_latencies_ms).
    """
    result: list[str] = []
    latencies: list[float] = []
    total = n_warmup + n_runs
    for i in range(total):
        t0 = time.perf_counter()
        result = fn(conn, table, keyword, limit)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            latencies.append(elapsed_ms)
    return result, latencies


def _timed_qdrant(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int,
    hnsw_ef: int | None,
    exact: bool,
    n_warmup: int,
    n_runs: int,
) -> tuple[list, list[float]]:
    """
    Run query_points n_warmup+n_runs times, discarding warm-up results.

    Times only the query_points call; embedding time is excluded because the
    vector is precomputed.  Returns (last_result_points, recorded_latencies_ms).
    """
    result = []
    latencies: list[float] = []
    total = n_warmup + n_runs
    for i in range(total):
        t0 = time.perf_counter()
        result = run_qdrant(client, collection, vector, limit, hnsw_ef, exact)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            latencies.append(elapsed_ms)
    return result, latencies


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recall(
    hnsw_ids: list[str],
    exact_ids: list[str],
    hnsw_source_ids: list[str],
    exact_source_ids: list[str],
) -> tuple[float, float]:
    """
    Compute strict and lenient recall@5 for one query.

    recall_at_5_by_point (strict):
        |{hnsw point ids} ∩ {exact point ids}| / |{exact point ids}|

    recall_at_5_by_source (lenient):
        |{hnsw source chunk ids} ∩ {exact source chunk ids}| / |{exact source chunk ids}|

    WHY TWO MEASURES: padded copies in the 10k and 50k datasets are
    near-duplicates of a real chunk (same text, tiny vector noise).  If HNSW
    returns a padded copy of a true nearest neighbour, the strict score counts
    it as a miss even though the retrieved text is semantically identical.
    The lenient score resolves both the HNSW hit and the exact hit back to their
    source chunk UUID before comparing, so near-duplicate hits are credited.
    Both columns are kept so the gap reveals how much dataset padding inflates
    apparent HNSW errors.
    """
    exact_set = set(exact_ids)
    exact_src_set = set(exact_source_ids)

    if not exact_set:
        return 1.0, 1.0  # no ground truth — treat as perfect

    hits_strict = sum(1 for pid in hnsw_ids if pid in exact_set)
    recall_strict = hits_strict / len(exact_set)

    if not exact_src_set:
        return recall_strict, recall_strict

    hits_lenient = sum(1 for src in hnsw_source_ids if src in exact_src_set)
    recall_lenient = hits_lenient / len(exact_src_set)

    return recall_strict, recall_lenient


def _point_ids_and_sources(points) -> tuple[list[str], list[str]]:
    """
    Extract (point_id_strings, source_chunk_id_strings) from a Qdrant result.

    source_chunk_id is set by generate_data.py: for real chunks it equals the
    chunk_id itself; for padded copies it is the UUID of the real chunk they
    were derived from.
    """
    ids: list[str] = []
    sources: list[str] = []
    for p in points:
        ids.append(str(p.id))
        payload = p.payload or {}
        # Fall back to the point id itself if the field is absent.
        sources.append(str(payload.get("source_chunk_id", p.id)))
    return ids, sources


# ---------------------------------------------------------------------------
# EXPLAIN helpers
# ---------------------------------------------------------------------------

def _run_explain(
    conn: psycopg2.extensions.connection,
    table: str,
    keyword: str,
) -> tuple[str, str]:
    """
    Run EXPLAIN (ANALYZE, BUFFERS) for one LIKE query and one FTS query on the
    given table.  Returns (like_plan_text, fts_plan_text).

    DBMS concept: EXPLAIN ANALYZE executes the query and shows the actual row
    counts, timing per plan node, and buffer (page) hit/miss statistics.
    This reveals whether the GIN index was used for FTS and whether the LIKE
    scan uses a sequential scan or a trigram index.
    """
    _assert_safe_table(table)
    plans = []
    for sql, params in [
        (
            f"""
            EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
            SELECT chunk_id FROM {table}
            WHERE chunk_text ILIKE %s
            ORDER BY chunk_id LIMIT %s
            """,
            (f"%{keyword}%", LIMIT),
        ),
        (
            f"""
            EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
            SELECT chunk_id FROM {table}
            WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(
                         to_tsvector('english', chunk_text),
                         plainto_tsquery('english', %s)
                     ) DESC, chunk_id LIMIT %s
            """,
            (keyword, keyword, LIMIT),
        ),
    ]:
        with conn.cursor() as cur:
            cur.execute(sql, params)  # noqa: S608 — table allowlisted
            rows = cur.fetchall()
        plans.append("\n".join(row[0] for row in rows))
    return plans[0], plans[1]


# ---------------------------------------------------------------------------
# Environment collector
# ---------------------------------------------------------------------------

def collect_environment(client: QdrantClient, conn: psycopg2.extensions.connection) -> str:
    """
    Gather version and hardware information for reproducibility.

    DBMS concept: recording the execution environment is essential for
    interpreting benchmark numbers.  Latency differences between runs on
    different hardware or software versions are not meaningful comparisons.
    """
    import datetime
    import multiprocessing

    lines: list[str] = [f"Date:            {datetime.datetime.now().isoformat(timespec='seconds')}"]

    # Python
    lines.append(f"Python:          {sys.version}")
    lines.append(f"OS:              {platform.platform()}")

    # CPU / RAM
    cpu_count = multiprocessing.cpu_count()
    cpu_name = platform.processor() or "unknown"
    lines.append(f"CPU name:        {cpu_name}")
    lines.append(f"CPU logical:     {cpu_count}")

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        lines.append(f"RAM (GB):        {ram_gb:.1f}")
    except ImportError:
        lines.append("RAM (GB):        psutil not installed")

    # Library versions
    try:
        import qdrant_client
        lines.append(f"qdrant-client:   {qdrant_client.__version__}")
    except AttributeError:
        lines.append("qdrant-client:   version unknown")

    try:
        import sentence_transformers
        lines.append(f"sentence-transformers: {sentence_transformers.__version__}")
    except AttributeError:
        lines.append("sentence-transformers: version unknown")

    # PostgreSQL server version
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            pg_ver = cur.fetchone()[0]
        lines.append(f"PostgreSQL:      {pg_ver}")
    except Exception as exc:
        lines.append(f"PostgreSQL:      error ({exc})")

    # Qdrant server version
    try:
        info = client.get_collections()  # lightweight call
        # Version is on the root info object
        ver_info = client._client.http.cluster_api.cluster_status()  # type: ignore[attr-defined]
        lines.append(f"Qdrant version:  {ver_info.result.version}")
    except Exception:
        # Fallback: version may be in the QdrantClient directly
        try:
            lines.append(f"Qdrant version:  (see Qdrant dashboard)")
        except Exception:
            lines.append("Qdrant version:  unknown")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(latencies_ms: list[float]) -> dict[str, float]:
    """Return p50, p95, mean rounded to 3 decimal places."""
    if not latencies_ms:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "mean_ms": float("nan")}
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "p50_ms":  round(float(np.percentile(arr, 50)), 3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_size(
    size: int,
    queries: list[dict[str, str]],
    vectors: list[list[float]],
    n_warmup: int,
    n_runs: int,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Run all five methods for one dataset size.

    Opens a fresh PostgreSQL connection and Qdrant client per size to avoid
    caching effects from earlier sizes bleeding into later ones.

    Returns (result_rows, like_explain_text, fts_explain_text) where
    result_rows is one dict per method suitable for results.csv.
    """
    cfg = SIZE_CONFIG[size]
    table = cfg["table"]
    collection = cfg["collection"]
    _assert_safe_table(table)
    _assert_safe_collection(collection)

    print(f"\n{'='*64}")
    print(f"  Size: {size:,}   table={table}   collection={collection}")
    print(f"{'='*64}")

    # --- Open connections ---
    try:
        pg_conn = _pg_connect()
    except Exception as exc:
        print(f"  ERROR: cannot connect to PostgreSQL: {exc}")
        return _all_failed_rows(size, n_runs, len(queries)), "ERROR", "ERROR"

    try:
        qd_client = _qdrant_client()
    except Exception as exc:
        print(f"  ERROR: cannot connect to Qdrant: {exc}")
        pg_conn.close()
        return _all_failed_rows(size, n_runs, len(queries)), "ERROR", "ERROR"

    # --- Collect per-(query, method) latency lists and recall values ---
    # Structure: method_name -> list[float] (all recorded runs across all queries)
    all_latencies: dict[str, list[float]] = {
        "sql_like":           [],
        "sql_fts":            [],
        "qdrant_hnsw_ef16":   [],
        "qdrant_hnsw_ef128":  [],
        "qdrant_exact":       [],
    }
    # recall values per query for the two HNSW variants
    recall_ef16_strict:   list[float] = []
    recall_ef16_lenient:  list[float] = []
    recall_ef128_strict:  list[float] = []
    recall_ef128_lenient: list[float] = []

    method_failed: dict[str, bool] = {m: False for m in all_latencies}

    for qi, query in enumerate(queries):
        qid      = query["id"]
        keyword  = query["keyword"]
        vector   = vectors[qi]
        print(f"  Query {qi+1}/{len(queries)}: [{qid}] keyword='{keyword}'")

        # ---- sql_like ----
        if not method_failed["sql_like"]:
            try:
                _, lats = _timed_sql(
                    pg_conn, table, run_sql_like, keyword, LIMIT, n_warmup, n_runs
                )
                all_latencies["sql_like"].extend(lats)
            except Exception as exc:
                print(f"    sql_like ERROR: {exc}")
                method_failed["sql_like"] = True

        # ---- sql_fts ----
        if not method_failed["sql_fts"]:
            try:
                _, lats = _timed_sql(
                    pg_conn, table, run_sql_fts, keyword, LIMIT, n_warmup, n_runs
                )
                all_latencies["sql_fts"].extend(lats)
            except Exception as exc:
                print(f"    sql_fts ERROR: {exc}")
                method_failed["sql_fts"] = True

        # ---- qdrant_exact (ground truth — run BEFORE HNSW so we have recall targets) ----
        exact_ids: list[str] = []
        exact_source_ids: list[str] = []
        if not method_failed["qdrant_exact"]:
            try:
                exact_points, lats = _timed_qdrant(
                    qd_client, collection, vector, LIMIT,
                    hnsw_ef=None, exact=True,
                    n_warmup=n_warmup, n_runs=n_runs,
                )
                all_latencies["qdrant_exact"].extend(lats)
                exact_ids, exact_source_ids = _point_ids_and_sources(exact_points)
            except Exception as exc:
                print(f"    qdrant_exact ERROR: {exc}")
                method_failed["qdrant_exact"] = True

        # ---- qdrant_hnsw_ef16 ----
        if not method_failed["qdrant_hnsw_ef16"]:
            try:
                ef16_points, lats = _timed_qdrant(
                    qd_client, collection, vector, LIMIT,
                    hnsw_ef=16, exact=False,
                    n_warmup=n_warmup, n_runs=n_runs,
                )
                all_latencies["qdrant_hnsw_ef16"].extend(lats)
                ef16_ids, ef16_src_ids = _point_ids_and_sources(ef16_points)
                rs, rl = compute_recall(ef16_ids, exact_ids, ef16_src_ids, exact_source_ids)
                recall_ef16_strict.append(rs)
                recall_ef16_lenient.append(rl)
            except Exception as exc:
                print(f"    qdrant_hnsw_ef16 ERROR: {exc}")
                method_failed["qdrant_hnsw_ef16"] = True

        # ---- qdrant_hnsw_ef128 ----
        if not method_failed["qdrant_hnsw_ef128"]:
            try:
                ef128_points, lats = _timed_qdrant(
                    qd_client, collection, vector, LIMIT,
                    hnsw_ef=128, exact=False,
                    n_warmup=n_warmup, n_runs=n_runs,
                )
                all_latencies["qdrant_hnsw_ef128"].extend(lats)
                ef128_ids, ef128_src_ids = _point_ids_and_sources(ef128_points)
                rs, rl = compute_recall(ef128_ids, exact_ids, ef128_src_ids, exact_source_ids)
                recall_ef128_strict.append(rs)
                recall_ef128_lenient.append(rl)
            except Exception as exc:
                print(f"    qdrant_hnsw_ef128 ERROR: {exc}")
                method_failed["qdrant_hnsw_ef128"] = True

    # --- EXPLAIN for the first keyword (representative sample) ---
    explain_like = "N/A"
    explain_fts  = "N/A"
    try:
        explain_like, explain_fts = _run_explain(pg_conn, table, queries[0]["keyword"])
    except Exception as exc:
        print(f"  EXPLAIN error: {exc}")

    pg_conn.close()

    # --- Aggregate statistics per method ---
    n_queries = len(queries)
    result_rows: list[dict[str, Any]] = []

    def _recall_avg(values: list[float]) -> float | str:
        return round(float(np.mean(values)), 4) if values else "failed"

    for method, lats in all_latencies.items():
        if method_failed[method]:
            row = {
                "size":                   size,
                "method":                 method,
                "p50_ms":                 "failed",
                "p95_ms":                 "failed",
                "mean_ms":                "failed",
                "recall_at_5_by_point":   "N/A",
                "recall_at_5_by_source":  "N/A",
                "n_queries":              n_queries,
                "n_runs":                 n_runs,
            }
        else:
            stats = _stats(lats)
            if method == "qdrant_hnsw_ef16":
                r_point  = _recall_avg(recall_ef16_strict)
                r_source = _recall_avg(recall_ef16_lenient)
            elif method == "qdrant_hnsw_ef128":
                r_point  = _recall_avg(recall_ef128_strict)
                r_source = _recall_avg(recall_ef128_lenient)
            elif method == "qdrant_exact":
                # Exact is ground truth; recall is 1.0 by definition.
                r_point  = 1.0
                r_source = 1.0
            else:
                r_point  = "N/A"  # SQL methods have no recall metric
                r_source = "N/A"

            row = {
                "size":                  size,
                "method":                method,
                **stats,
                "recall_at_5_by_point":  r_point,
                "recall_at_5_by_source": r_source,
                "n_queries":             n_queries,
                "n_runs":                n_runs,
            }

        result_rows.append(row)
        _print_row(row)

    return result_rows, explain_like, explain_fts


def _all_failed_rows(size: int, n_runs: int, n_queries: int) -> list[dict[str, Any]]:
    """Return one 'failed' row per method when the whole size is unreachable."""
    methods = ["sql_like", "sql_fts", "qdrant_hnsw_ef16", "qdrant_hnsw_ef128", "qdrant_exact"]
    return [
        {
            "size":                  size,
            "method":                m,
            "p50_ms":                "failed",
            "p95_ms":                "failed",
            "mean_ms":               "failed",
            "recall_at_5_by_point":  "N/A",
            "recall_at_5_by_source": "N/A",
            "n_queries":             n_queries,
            "n_runs":                n_runs,
        }
        for m in methods
    ]


def _print_row(row: dict[str, Any]) -> None:
    """Print a single result row in a compact readable format."""
    p50 = row["p50_ms"]
    p95 = row["p95_ms"]
    rp  = row["recall_at_5_by_point"]
    rs  = row["recall_at_5_by_source"]
    p50_str = f"{p50:>8.2f}" if isinstance(p50, float) else f"{'failed':>8}"
    p95_str = f"{p95:>8.2f}" if isinstance(p95, float) else f"{'failed':>8}"
    rp_str  = f"{rp:.4f}" if isinstance(rp, float) else str(rp)
    rs_str  = f"{rs:.4f}" if isinstance(rs, float) else str(rs)
    print(
        f"    {row['method']:<22} p50={p50_str} ms  p95={p95_str} ms"
        f"  recall_pt={rp_str}  recall_src={rs_str}"
    )


# ---------------------------------------------------------------------------
# CSV / file writers
# ---------------------------------------------------------------------------

RESULTS_FIELDNAMES = [
    "size", "method",
    "p50_ms", "p95_ms", "mean_ms",
    "recall_at_5_by_point", "recall_at_5_by_source",
    "n_queries", "n_runs",
]


def save_results_csv(all_rows: list[dict[str, Any]]) -> None:
    """
    Write results.csv, overwriting any existing file so there are never duplicate rows.

    DBMS concept: idempotent output is important; a simple overwrite avoids the
    need for a 'delete where run_id = X' query that OLAP tools sometimes require.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults written to {path}")


def save_embedding_times_csv(embedding_times: dict[str, float]) -> None:
    """Write embedding_times.csv with one row per query."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "embedding_times.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["query_id", "embedding_ms"])
        writer.writeheader()
        for qid, ms in embedding_times.items():
            writer.writerow({"query_id": qid, "embedding_ms": round(ms, 3)})
    print(f"Embedding times written to {path}")


def save_explain(size: int, like_plan: str, fts_plan: str) -> None:
    """Write EXPLAIN (ANALYZE, BUFFERS) output for one size to explain_<size>.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"explain_{size}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"=== LIKE query (size={size}) ===\n\n")
        fh.write(like_plan)
        fh.write(f"\n\n=== FTS query (size={size}) ===\n\n")
        fh.write(fts_plan)
        fh.write("\n")
    print(f"EXPLAIN plan written to {path}")


def save_environment(client: QdrantClient, conn: psycopg2.extensions.connection) -> None:
    """Write environment.txt with version and hardware details."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "environment.txt"
    env_text = collect_environment(client, conn)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(env_text)
        fh.write("\n")
    print(f"Environment info written to {path}")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def save_graphs(all_rows: list[dict[str, Any]]) -> None:
    """
    Generate two PNG graphs from results.csv data:

      1. latency_vs_size.png – p50 latency per method per size on a log y-axis,
         with p95 as a second series (dashed).
      2. recall_vs_latency.png – scatter of ef16 vs ef128 at each size,
         showing the recall/latency trade-off.

    DBMS concept: visualising query plans and latency at scale is standard
    database performance engineering practice.  Log-scale y-axis is used
    because HNSW latency (< 10 ms) and ILIKE latency (> 100 ms at 50 k) differ
    by orders of magnitude; linear scale would hide HNSW differences.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping graph generation.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Pivot data: method -> size -> stats
    methods = ["sql_like", "sql_fts", "qdrant_hnsw_ef16", "qdrant_hnsw_ef128", "qdrant_exact"]
    sizes = sorted(SIZE_CONFIG.keys())

    def _safe_float(val: Any) -> float | None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    pivoted: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in methods}
    for row in all_rows:
        m = row["method"]
        s = int(row["size"])
        if m in pivoted:
            pivoted[m][s] = row

    colors = {
        "sql_like":           "#e74c3c",
        "sql_fts":            "#e67e22",
        "qdrant_hnsw_ef16":   "#2980b9",
        "qdrant_hnsw_ef128":  "#27ae60",
        "qdrant_exact":       "#8e44ad",
    }

    # ---- Graph 1: latency_vs_size ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        p50_vals = [_safe_float(pivoted[method].get(s, {}).get("p50_ms")) for s in sizes]
        p95_vals = [_safe_float(pivoted[method].get(s, {}).get("p95_ms")) for s in sizes]
        valid = [(s, p, q) for s, p, q in zip(sizes, p50_vals, p95_vals) if p is not None]
        if not valid:
            continue
        xs, p50s, p95s = zip(*valid)
        c = colors.get(method, "gray")
        ax.plot(xs, p50s, marker="o", label=f"{method} p50", color=c)
        ax.plot(xs, p95s, marker="s", linestyle="--", label=f"{method} p95", color=c, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Dataset size (rows)")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Query latency vs. dataset size — p50 (solid) and p95 (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out1 = RESULTS_DIR / "latency_vs_size.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Graph written to {out1}")

    # ---- Graph 2: recall_vs_latency ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, label, marker in [
        ("qdrant_hnsw_ef16",  "hnsw_ef=16",  "o"),
        ("qdrant_hnsw_ef128", "hnsw_ef=128", "s"),
    ]:
        for s in sizes:
            row = pivoted[method].get(s)
            if not row:
                continue
            mean_ms = _safe_float(row.get("mean_ms"))
            recall  = _safe_float(row.get("recall_at_5_by_point"))
            if mean_ms is None or recall is None:
                continue
            ax.scatter(mean_ms, recall, marker=marker, s=80,
                       label=f"{label} @ {s:,}" if s == sizes[0] else "")
            ax.annotate(f"{s//1000}k", (mean_ms, recall),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

    # Add legend handles manually to keep it clean
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="hnsw_ef=16"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=8, label="hnsw_ef=128"),
    ]
    ax.legend(handles=handles)
    ax.set_xlabel("Mean query latency (ms)")
    ax.set_ylabel("Recall@5 (by point id)")
    ax.set_title("HNSW recall vs. latency trade-off (ef=16 vs ef=128)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out2 = RESULTS_DIR / "recall_vs_latency.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Graph written to {out2}")


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary(all_rows: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table at the end of the run."""
    header = (
        f"\n{'Size':>7}  {'Method':<22}  {'p50 ms':>8}  {'p95 ms':>8}"
        f"  {'mean ms':>8}  {'R@5 pt':>7}  {'R@5 src':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in all_rows:
        p50 = row["p50_ms"]
        p95 = row["p95_ms"]
        mn  = row["mean_ms"]
        rp  = row["recall_at_5_by_point"]
        rs  = row["recall_at_5_by_source"]
        p50_s = f"{p50:>8.2f}" if isinstance(p50, float) else f"{'failed':>8}"
        p95_s = f"{p95:>8.2f}" if isinstance(p95, float) else f"{'failed':>8}"
        mn_s  = f"{mn:>8.2f}"  if isinstance(mn, float)  else f"{'failed':>8}"
        rp_s  = f"{float(rp):>7.4f}" if isinstance(rp, float) else f"{'N/A':>7}"
        rs_s  = f"{float(rs):>8.4f}" if isinstance(rs, float) else f"{'N/A':>8}"
        print(
            f"{row['size']:>7,}  {row['method']:<22}  {p50_s}  {p95_s}"
            f"  {mn_s}  {rp_s}  {rs_s}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
            "Dataset sizes to benchmark. "
            f"Allowed: {sorted(SIZE_CONFIG.keys())}. "
            "Default: all three."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 3 queries with 2 recorded runs each — fast sanity check.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load queries, embed, benchmark each size, write outputs."""
    args = _parse_args()
    sizes = sorted(args.sizes)
    n_runs = 2 if args.quick else N_RUNS

    print(f"\n{'='*64}")
    print(f"  Semantic Notes Search — Benchmark Runner")
    print(f"  Sizes: {sizes}   quick={args.quick}   n_runs={n_runs}")
    print(f"{'='*64}")

    # Load queries
    queries = load_queries(QUERIES_FILE, quick=args.quick)
    print(f"Loaded {len(queries)} queries from {QUERIES_FILE}.")

    # Embed all questions once — outside every timed section.
    vectors, embedding_times = embed_queries(queries)

    all_rows: list[dict[str, Any]] = []

    # Benchmark each size in turn.
    for size in sizes:
        rows, explain_like, explain_fts = run_size(
            size=size,
            queries=queries,
            vectors=vectors,
            n_warmup=N_WARMUP,
            n_runs=n_runs,
        )
        all_rows.extend(rows)
        save_explain(size, explain_like, explain_fts)

    # Write CSV outputs.
    save_results_csv(all_rows)
    save_embedding_times_csv(embedding_times)

    # Write environment info (reuse a connection from any size that worked).
    try:
        env_conn   = _pg_connect()
        env_client = _qdrant_client()
        save_environment(env_client, env_conn)
        env_conn.close()
    except Exception as exc:
        print(f"WARNING: could not write environment.txt: {exc}")

    # Generate graphs.
    save_graphs(all_rows)

    # Print summary table.
    print_summary(all_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
