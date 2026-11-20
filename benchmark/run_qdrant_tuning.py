"""
Qdrant client tuning benchmark for the Semantic Notes Search Engine.

Evaluates the performance impact of network transport and hostname resolution
on Qdrant vector search across three dataset sizes (1k, 10k, 50k):

  Client setups compared:
    1. http_localhost – HTTP client connecting to "localhost" (port 6333).
                        Windows often incurs DNS lookup latency / IPv6 fallback
                        delays when resolving "localhost".
    2. http_127_0_0_1 – HTTP client connecting to literal "127.0.0.1" (port 6333),
                        bypassing Windows hostname resolution.
    3. grpc           – gRPC client using HTTP/2 multiplexing (prefer_grpc=True,
                        port 6334) with binary protobuf serialization.

  Search methods re-timed:
    - qdrant_exact       (SearchParams(exact=True))
    - qdrant_hnsw_ef16   (SearchParams(hnsw_ef=16, exact=False))
    - qdrant_hnsw_ef128  (SearchParams(hnsw_ef=128, exact=False))

  Timing methodology:
    - Load SentenceTransformer once, pre-embed all 20 queries outside timed sections.
    - Warm-up pass per size and setup before recording.
    - 30 recorded runs per query and method (600 runs per condition).
    - Client-side latency measured via time.perf_counter().
    - Server-side search execution time extracted from Qdrant response if provided.
    - Results saved to benchmark/results/qdrant_tuning.csv.

Usage:
    py benchmark/run_qdrant_tuning.py
    py benchmark/run_qdrant_tuning.py --quick
    py benchmark/run_qdrant_tuning.py --sizes 1000 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
OUTPUT_CSV = RESULTS_DIR / "qdrant_tuning.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SIZE_CONFIG: dict[int, str] = {
    1_000: "bench_1k",
    10_000: "bench_10k",
    50_000: "bench_50k",
}

ALLOWED_COLLECTIONS = set(SIZE_CONFIG.values())

METHODS = [
    "qdrant_exact",
    "qdrant_hnsw_ef16",
    "qdrant_hnsw_ef128",
]

DEFAULT_REPEATS = 30
DEFAULT_WARMUP = 2
LIMIT = 5

load_dotenv(PROJECT_ROOT / ".env")


def _get_api_key() -> str | None:
    return os.getenv("QDRANT_API_KEY") or None


def _assert_safe_collection(collection: str) -> None:
    """Reject any collection name not in the bench allowlist."""
    if collection not in ALLOWED_COLLECTIONS:
        raise ValueError(
            f"Refusing to query collection '{collection}'. "
            f"Only bench collections allowed: {sorted(ALLOWED_COLLECTIONS)}"
        )


# ---------------------------------------------------------------------------
# Client Setups
# ---------------------------------------------------------------------------

def init_clients() -> dict[str, QdrantClient]:
    """
    Initialize and reuse one client instance per setup.

    Setup (a): http_localhost – HTTP client to 'localhost'
    Setup (b): http_127_0_0_1 – HTTP client to '127.0.0.1'
    Setup (c): grpc           – gRPC client with prefer_grpc=True
    """
    api_key = _get_api_key()

    clients = {
        "http_localhost": QdrantClient(
            url="http://localhost:6333",
            api_key=api_key,
            prefer_grpc=False,
            timeout=10,
        ),
        "http_127_0_0_1": QdrantClient(
            url="http://127.0.0.1:6333",
            api_key=api_key,
            prefer_grpc=False,
            timeout=10,
        ),
        "grpc": QdrantClient(
            host="127.0.0.1",
            port=6333,
            grpc_port=6334,
            api_key=api_key,
            prefer_grpc=True,
            timeout=10,
        ),
    }
    return clients


# ---------------------------------------------------------------------------
# Query & Embedding Loading
# ---------------------------------------------------------------------------

def load_queries(path: Path, quick: bool) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        queries = json.load(fh)
    if quick:
        queries = queries[:3]
    return queries


def embed_queries(queries: list[dict[str, str]]) -> list[list[float]]:
    """Load model once and embed all queries upfront (untimed)."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'…")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    questions = [q["question"] for q in queries]

    t0 = time.perf_counter()
    vectors_np = model.encode(
        questions,
        show_progress_bar=False,
        normalize_embeddings=True,
        batch_size=len(questions),
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Embedded {len(queries)} questions in {total_ms:.1f} ms.")
    return [v.tolist() for v in vectors_np]


# ---------------------------------------------------------------------------
# Search Execution & Timing
# ---------------------------------------------------------------------------

def get_search_params(method: str) -> SearchParams:
    if method == "qdrant_exact":
        return SearchParams(exact=True)
    elif method == "qdrant_hnsw_ef16":
        return SearchParams(hnsw_ef=16, exact=False)
    elif method == "qdrant_hnsw_ef128":
        return SearchParams(hnsw_ef=128, exact=False)
    else:
        raise ValueError(f"Unknown method: {method}")


def execute_query(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    search_params: SearchParams,
) -> tuple[float, float | None]:
    """
    Execute a single query_points call.

    Returns:
        (client_time_ms, server_time_ms_or_None)
    """
    t0 = time.perf_counter()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        search_params=search_params,
        limit=LIMIT,
        with_payload=False,
    )
    client_ms = (time.perf_counter() - t0) * 1000.0

    # Extract server-side execution time if provided by response
    server_ms = None
    server_time_s = getattr(response, "time", None)
    if server_time_s is None and hasattr(response, "model_extra") and response.model_extra:
        server_time_s = response.model_extra.get("time")

    if server_time_s is not None:
        try:
            server_ms = float(server_time_s) * 1000.0
        except (ValueError, TypeError):
            server_ms = None

    return client_ms, server_ms


# ---------------------------------------------------------------------------
# Statistics Helper
# ---------------------------------------------------------------------------

def _calc_stats(values: list[float]) -> tuple[float, float, float]:
    """Return (p50, p95, mean) in ms."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values, dtype=np.float64)
    return (
        round(float(np.percentile(arr, 50)), 3),
        round(float(np.percentile(arr, 95)), 3),
        round(float(np.mean(arr)), 3),
    )


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "size",
    "setup",
    "method",
    "client_p50_ms",
    "client_p95_ms",
    "client_mean_ms",
    "server_p50_ms",
    "server_p95_ms",
    "n_queries",
    "n_repeats",
]


def run_tuning(
    sizes: list[int],
    queries: list[dict[str, str]],
    vectors: list[list[float]],
    clients: dict[str, QdrantClient],
    n_warmup: int,
    n_repeats: int,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []

    for size in sizes:
        collection = SIZE_CONFIG[size]
        _assert_safe_collection(collection)

        print(f"\n{'='*70}")
        print(f"Dataset Size: {size:,} (collection={collection})")
        print(f"{'='*70}")

        for setup_name, client in clients.items():
            print(f"\n--- Client setup: {setup_name} ---")

            # Warm-up pass on first 3 queries
            warmup_queries = queries[:3]
            for qi in range(len(warmup_queries)):
                vec = vectors[qi]
                for method in METHODS:
                    sp = get_search_params(method)
                    for _ in range(n_warmup):
                        try:
                            execute_query(client, collection, vec, sp)
                        except Exception:
                            pass

            # Measurement runs
            for method in METHODS:
                sp = get_search_params(method)
                client_latencies: list[float] = []
                server_latencies: list[float] = []
                failed = False

                for qi, query in enumerate(queries):
                    vec = vectors[qi]
                    for _ in range(n_repeats):
                        try:
                            c_ms, s_ms = execute_query(client, collection, vec, sp)
                            client_latencies.append(c_ms)
                            if s_ms is not None:
                                server_latencies.append(s_ms)
                        except Exception as exc:
                            print(f"    ERROR [{setup_name} | {method} | {query['id']}]: {exc}")
                            failed = True
                            break
                    if failed:
                        break

                if failed or not client_latencies:
                    row = {
                        "size": size,
                        "setup": setup_name,
                        "method": method,
                        "client_p50_ms": "failed",
                        "client_p95_ms": "failed",
                        "client_mean_ms": "failed",
                        "server_p50_ms": "N/A",
                        "server_p95_ms": "N/A",
                        "n_queries": len(queries),
                        "n_repeats": n_repeats,
                    }
                else:
                    cp50, cp95, cmean = _calc_stats(client_latencies)
                    if server_latencies:
                        sp50, sp95, _ = _calc_stats(server_latencies)
                        s_p50_str: float | str = sp50
                        s_p95_str: float | str = sp95
                    else:
                        s_p50_str = "N/A"
                        s_p95_str = "N/A"

                    row = {
                        "size": size,
                        "setup": setup_name,
                        "method": method,
                        "client_p50_ms": cp50,
                        "client_p95_ms": cp95,
                        "client_mean_ms": cmean,
                        "server_p50_ms": s_p50_str,
                        "server_p95_ms": s_p95_str,
                        "n_queries": len(queries),
                        "n_repeats": n_repeats,
                    }

                all_rows.append(row)
                _print_row(row)

    return all_rows


def _print_row(row: dict[str, Any]) -> None:
    cp50 = row["client_p50_ms"]
    cp95 = row["client_p95_ms"]
    sp50 = row["server_p50_ms"]
    cp50_s = f"{cp50:>7.2f}" if isinstance(cp50, float) else f"{cp50:>7}"
    cp95_s = f"{cp95:>7.2f}" if isinstance(cp95, float) else f"{cp95:>7}"
    sp50_s = f"{sp50:>7.2f}" if isinstance(sp50, float) else f"{sp50:>7}"
    print(
        f"  {row['method']:<20} client_p50={cp50_s} ms  "
        f"client_p95={cp95_s} ms  server_p50={sp50_s} ms"
    )


def save_csv(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved tuning results to {OUTPUT_CSV}")


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"\n{'Size':>6}  {'Setup':<16}  {'Method':<20}  "
        f"{'c_p50 ms':>9}  {'c_p95 ms':>9}  {'c_mean ms':>10}  "
        f"{'s_p50 ms':>9}  {'s_p95 ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        cp50 = r["client_p50_ms"]
        cp95 = r["client_p95_ms"]
        cmean = r["client_mean_ms"]
        sp50 = r["server_p50_ms"]
        sp95 = r["server_p95_ms"]

        cp50_s = f"{cp50:>9.2f}" if isinstance(cp50, float) else f"{cp50:>9}"
        cp95_s = f"{cp95:>9.2f}" if isinstance(cp95, float) else f"{cp95:>9}"
        cmean_s = f"{cmean:>10.2f}" if isinstance(cmean, float) else f"{cmean:>10}"
        sp50_s = f"{sp50:>9.2f}" if isinstance(sp50, float) else f"{sp50:>9}"
        sp95_s = f"{sp95:>9.2f}" if isinstance(sp95, float) else f"{sp95:>9}"

        print(
            f"{r['size']:>6,}  {r['setup']:<16}  {r['method']:<20}  "
            f"{cp50_s}  {cp95_s}  {cmean_s}  {sp50_s}  {sp95_s}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
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
        help="Dataset sizes to benchmark. Default: all 3 (1000, 10000, 50000).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Recorded runs per query per method. Default: {DEFAULT_REPEATS}.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 3 queries with 2 repeats each for a fast sanity check.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sizes = sorted(args.sizes)
    n_repeats = 2 if args.quick else args.repeats
    n_warmup = 1 if args.quick else DEFAULT_WARMUP

    print(f"\n{'='*70}")
    print(f"  Qdrant Client Tuning Benchmark")
    print(f"  Sizes: {sizes} | Repeats: {n_repeats} | Quick: {args.quick}")
    print(f"{'='*70}")

    queries = load_queries(QUERIES_FILE, quick=args.quick)
    vectors = embed_queries(queries)

    clients = init_clients()
    try:
        rows = run_tuning(
            sizes=sizes,
            queries=queries,
            vectors=vectors,
            clients=clients,
            n_warmup=n_warmup,
            n_repeats=n_repeats,
        )
    finally:
        for c in clients.values():
            try:
                c.close()
            except Exception:
                pass

    save_csv(rows)
    print_table(rows)
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
