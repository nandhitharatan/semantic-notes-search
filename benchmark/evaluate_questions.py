"""
Comparative evaluation runner with subject-filter comparison.

Evaluates the 20 benchmark questions across four retrieval methods:
  1. SQL LIKE            (substring scan, no filter)
  2. SQL FTS             (GIN full-text search, no filter)
  3. Qdrant Vector       (cosine semantic search, no subject filter)
  4. Qdrant Vector+Filter (cosine semantic search, filtered by question subject)

HIT DEFINITION:
  A retrieval method scores a "hit" if and only if the ground truth `expected_file`
  AND `expected_page` appear within its top-5 returned results.
  If a question is missing `expected_file` or `expected_page`, a warning is printed
  and that question is excluded from hit-rate calculations (not counted as a miss).

PAGE CONVENTION:
  1-based indexing (Page 1 is the first page of the PDF).

Usage:
    py benchmark/evaluate_questions.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the core search implementations directly without rewriting logic
from src.search_sql import search_fts, search_like
from src.search_vector import search_vector

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
QUESTIONS_FILE = BENCHMARK_DIR / "questions.json"
OUTPUT_CSV = RESULTS_DIR / "questions_results.csv"

TOP_K = 5
PAGE_NUMBER_CONVENTION = "1-based (Page 1 is the first page of the PDF)"

load_dotenv(PROJECT_ROOT / ".env")


def _pg_connect() -> psycopg2.extensions.connection:
    """Open PostgreSQL connection from .env."""
    url = os.getenv("POSTGRES_URL")
    if not url:
        raise RuntimeError("Missing POSTGRES_URL in environment.")
    return psycopg2.connect(url)


def _enrich_vector_results(
    pg_conn: psycopg2.extensions.connection,
    vector_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Look up file_name and page_no from PostgreSQL for each vector result.

    DBMS concept: In a polyglot persistence architecture, Qdrant stores the dense
    vector index while PostgreSQL maintains the authoritative relational catalog
    (documents and chunks). We join on doc_id and chunk_text to attach exact PDF
    provenance (file_name, page_no) to each vector hit.
    """
    enriched = []
    with pg_conn.cursor() as cur:
        for r in vector_results:
            cur.execute(
                """
                SELECT d.file_name, c.page_no
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.doc_id = %s AND c.chunk_text = %s
                LIMIT 1
                """,
                (r.get("doc_id"), r.get("text", "")),
            )
            row = cur.fetchone()
            file_name = row[0] if row else "-"
            page_no = row[1] if row else "-"
            enriched.append(
                {
                    "file_name": file_name,
                    "page_no": page_no,
                    "chunk_text": r.get("text", ""),
                    "score": r.get("score", 0.0),
                }
            )
    return enriched


def _normalise_expected_pages(raw: Any) -> list[int]:
    """
    Normalise expected_page to a list of ints so the hit check handles:
      - int            5          -> [5]
      - numeric string "5"        -> [5]
      - list of ints   [5, 6]     -> [5, 6]
      - list of strs   ["5","6"]  -> [5, 6]
      - empty string / empty list -> []  (means "no ground truth")
    """
    if raw is None or raw == "" or raw == []:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw if str(x).strip() != ""]
    # int or numeric string
    s = str(raw).strip()
    if s == "":
        return []
    return [int(s)]


def check_hit(results: list[dict[str, Any]], expected_file: str, expected_pages: list[int]) -> bool:
    """
    Determine if the ground-truth document and page appear within the top-5 results.

    A hit is True if and only if:
      - file_name matches expected_file (case-insensitive), AND
      - page_no (cast to int) is in the expected_pages list
    for at least one result in the top-5.

    expected_pages is already normalised to a list of ints by _normalise_expected_pages().
    An empty list means "no ground truth" and always returns False.
    """
    if not expected_file or not expected_pages:
        return False

    exp_f = expected_file.strip().lower()

    for r in results:
        rf = str(r.get("file_name", "")).strip().lower()
        try:
            rp = int(str(r.get("page_no", "")).strip())
        except (ValueError, TypeError):
            continue
        if rf == exp_f and rp in expected_pages:
            return True
    return False


def _clean_snippet(text: str, max_len: int = 120) -> str:
    """Normalize whitespace and truncate long chunk text for readable terminal printing."""
    cleaned = " ".join(text.split())
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "…"
    return cleaned


def main() -> None:
    print(f"\n{'='*78}")
    print(f"  Semantic Notes Search — Subject-Filtered Headline Questions Evaluation")
    print(f"{'='*78}")
    print(f"Page-number convention : {PAGE_NUMBER_CONVENTION}")
    print(f"Dataset / Collection   : chunks (PostgreSQL) & notes_chunks (Qdrant)")
    print(f"Retrieval Top-K        : {TOP_K}")
    print(f"{'='*78}\n")

    with open(QUESTIONS_FILE, encoding="utf-8") as fh:
        questions: list[dict[str, Any]] = json.load(fh)

    pg_conn = _pg_connect()
    records: list[dict[str, Any]] = []

    print("Evaluating Questions & Top-1 Snippets:")
    print("-" * 78)

    for item in questions:
        qid = item["id"]
        qtype = item.get("query_type", "keyword")
        query_text = item["query"]
        subject = item.get("subject", "")

        expected_f = str(item.get("expected_file", "")).strip()
        # Normalise expected_page: accept int, numeric string, or list -> list[int]
        expected_pages = _normalise_expected_pages(item.get("expected_page", ""))
        has_ground_truth = bool(expected_f and expected_pages)
        # For display / CSV storage keep a compact string form
        expected_p_str = ",".join(str(p) for p in expected_pages) if expected_pages else ""

        if not has_ground_truth:
            print(f"\n[WARNING] Query {qid} ({qtype}): missing expected_file or expected_page. Excluded from hit rate calculations.")
        else:
            print(f"\nQuery {qid} ({qtype}) [Expected: {expected_f} p.{expected_p_str}]")

        print(f"  Subject: {subject} | Query: \"{query_text}\"")

        # 1. SQL LIKE (no filter)
        raw_like = search_like(query_text, limit=TOP_K)
        like_hit = check_hit(raw_like, expected_f, expected_pages)

        # 2. SQL FTS (no filter)
        raw_fts = search_fts(query_text, limit=TOP_K)
        fts_hit = check_hit(raw_fts, expected_f, expected_pages)

        # 3. Vector Search (no filter)
        raw_vec = search_vector(query_text, limit=TOP_K)
        vec_results = _enrich_vector_results(pg_conn, raw_vec)
        vec_hit = check_hit(vec_results, expected_f, expected_pages)

        # 4. Vector Search (with subject filter)
        raw_vec_flt = search_vector(query_text, limit=TOP_K, subject=subject)
        vec_flt_results = _enrich_vector_results(pg_conn, raw_vec_flt)
        vec_flt_hit = check_hit(vec_flt_results, expected_f, expected_pages)

        # Top-1 extractions
        top_like = raw_like[0] if raw_like else {}
        top_fts = raw_fts[0] if raw_fts else {}
        top_vec = vec_results[0] if vec_results else {}
        top_vec_flt = vec_flt_results[0] if vec_flt_results else {}

        like_file = top_like.get("file_name", "-")
        like_page = top_like.get("page_no", "-")
        like_snip = _clean_snippet(top_like.get("chunk_text", "[No results]"))

        fts_file = top_fts.get("file_name", "-")
        fts_page = top_fts.get("page_no", "-")
        fts_snip = _clean_snippet(top_fts.get("chunk_text", "[No results]"))

        vec_file = top_vec.get("file_name", "-")
        vec_page = top_vec.get("page_no", "-")
        vec_score = round(float(top_vec.get("score", 0.0)), 4) if top_vec else 0.0
        vec_snip = _clean_snippet(top_vec.get("chunk_text", "[No results]"))

        vec_flt_file = top_vec_flt.get("file_name", "-")
        vec_flt_page = top_vec_flt.get("page_no", "-")
        vec_flt_score = round(float(top_vec_flt.get("score", 0.0)), 4) if top_vec_flt else 0.0
        vec_flt_snip = _clean_snippet(top_vec_flt.get("chunk_text", "[No results]"))

        # Print top-1 inspection details
        print(f"    LIKE top-1          : [{like_file} p.{like_page}] {like_snip}")
        print(f"    FTS top-1           : [{fts_file} p.{fts_page}] {fts_snip}")
        print(f"    Vector top-1        : [{vec_file} p.{vec_page}] (score={vec_score:.4f}) {vec_snip}")
        print(f"    Vector+Filter top-1 : [{vec_flt_file} p.{vec_flt_page}] (score={vec_flt_score:.4f}) {vec_flt_snip}")

        records.append(
            {
                "id": qid,
                "subject": subject,
                "query_type": qtype,
                "query": query_text,
                "expected_file": expected_f,
                "expected_page": expected_p_str,
                "has_ground_truth": has_ground_truth,
                "like_hit": like_hit,
                "fts_hit": fts_hit,
                "vector_hit": vec_hit,
                "vector_filter_hit": vec_flt_hit,
                "like_top_file": like_file,
                "like_top_page": like_page,
                "fts_top_file": fts_file,
                "fts_top_page": fts_page,
                "vector_top_file": vec_file,
                "vector_top_page": vec_page,
                "vector_top_score": vec_score,
                "vector_filter_top_file": vec_flt_file,
                "vector_filter_top_page": vec_flt_page,
                "vector_filter_top_score": vec_flt_score,
                "vector_top_snippet": vec_snip,
                "vector_filter_top_snippet": vec_flt_snip,
            }
        )

    pg_conn.close()

    # ---------------------------------------------------------------------------
    # Table of Results (id, group, LIKE hit, FTS hit, Vector hit, Vector+filter hit)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Method Comparison Table:")
    print("=" * 78)
    header = f"{'ID':<10}  {'Group':<12}  {'LIKE':<8}  {'FTS':<8}  {'Vector':<8}  {'Vector+Filter':<14}"
    print(header)
    print("-" * len(header))

    def _fmt_hit(hit: bool, has_gt: bool) -> str:
        if not has_gt:
            return "N/A"
        return "Y" if hit else "N"

    for r in records:
        gt = r["has_ground_truth"]
        l_str = _fmt_hit(r["like_hit"], gt)
        f_str = _fmt_hit(r["fts_hit"], gt)
        v_str = _fmt_hit(r["vector_hit"], gt)
        vf_str = _fmt_hit(r["vector_filter_hit"], gt)
        print(f"{r['id']:<10}  {r['query_type']:<12}  {l_str:<8}  {f_str:<8}  {v_str:<8}  {vf_str:<14}")

    # ---------------------------------------------------------------------------
    # Hit Rates Breakdown
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Hit Rate Summary (Only verified questions with ground-truth are counted):")
    print("=" * 78)

    def _calc_hit_rates(subset: list[dict[str, Any]], group_name: str) -> None:
        valid_items = [item for item in subset if item["has_ground_truth"]]
        n_valid = len(valid_items)
        n_total = len(subset)

        print(f"\n{group_name} ({n_valid}/{n_total} questions have ground truth set):")
        if n_valid == 0:
            print("  [All questions currently missing expected_file / expected_page — hit rates excluded]")
            return

        l_cnt = sum(1 for item in valid_items if item["like_hit"])
        f_cnt = sum(1 for item in valid_items if item["fts_hit"])
        v_cnt = sum(1 for item in valid_items if item["vector_hit"])
        vf_cnt = sum(1 for item in valid_items if item["vector_filter_hit"])

        print(f"  SQL LIKE Hit Rate          : {l_cnt}/{n_valid} ({l_cnt / n_valid * 100:.1f}%)")
        print(f"  SQL FTS Hit Rate           : {f_cnt}/{n_valid} ({f_cnt / n_valid * 100:.1f}%)")
        print(f"  Vector Hit Rate            : {v_cnt}/{n_valid} ({v_cnt / n_valid * 100:.1f}%)")
        print(f"  Vector+Filter Hit Rate     : {vf_cnt}/{n_valid} ({vf_cnt / n_valid * 100:.1f}%)")

    kw_records = [r for r in records if r["query_type"] == "keyword"]
    para_records = [r for r in records if r["query_type"] == "paraphrase"]

    _calc_hit_rates(kw_records, "1. Keyword-Style Group")
    _calc_hit_rates(para_records, "2. Paraphrase Group")
    _calc_hit_rates(records, "3. Overall")

    # ---------------------------------------------------------------------------
    # Paraphrase Subject-Filter Impact
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("Paraphrase Group — Subject Filter Impact Analysis:")
    print("=" * 78)

    valid_para = [r for r in para_records if r["has_ground_truth"]]
    if not valid_para:
        print("Notice: No paraphrase questions currently have expected_file/expected_page set.")
        print("When ground-truth is filled:")
        print("  - Fixed questions : miss without filter (Vector=N), hit with filter (Vector+Filter=Y)")
        print("  - Broken questions: hit without filter (Vector=Y), miss with filter (Vector+Filter=N)")
    else:
        fixed = [r["id"] for r in valid_para if (not r["vector_hit"] and r["vector_filter_hit"])]
        broken = [r["id"] for r in valid_para if (r["vector_hit"] and not r["vector_filter_hit"])]

        print(f"  Filter FIXED ({len(fixed)}): {fixed if fixed else 'None'}")
        print(f"  Filter BROKE ({len(broken)}): {broken if broken else 'None'}")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "subject",
        "query_type",
        "query",
        "expected_file",
        "expected_page",
        "has_ground_truth",
        "like_hit",
        "fts_hit",
        "vector_hit",
        "vector_filter_hit",
        "like_top_file",
        "like_top_page",
        "fts_top_file",
        "fts_top_page",
        "vector_top_file",
        "vector_top_page",
        "vector_top_score",
        "vector_filter_top_file",
        "vector_filter_top_page",
        "vector_filter_top_score",
        "vector_top_snippet",
        "vector_filter_top_snippet",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
