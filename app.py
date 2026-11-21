"""
Semantic Notes Search Engine — Streamlit front-end.

Three retrieval methods side-by-side:
  1. SQL LIKE   (red header)   — search_like()  from src/search_sql.py
  2. SQL FTS    (yellow header)— search_fts()   from src/search_sql.py
  3. Vector     (green header) — search_vector() from src/search_vector.py
                                 (we override the Qdrant client here with gRPC
                                  for lower latency; see the note in _vec_client())

Sidebar lets you upload a PDF and ingest it via ingest_pdf() from src/ingest.py.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import psycopg2
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Path & env setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Reuse the actual search and ingest logic — do not rewrite it.
from src.search_sql import search_fts, search_like
from src.ingest import ingest_pdf

COLLECTION_NAME = "notes_chunks"
TOP_K = 5           # results to show per method
SNIPPET_LEN = 200   # characters of chunk text to display

# ---------------------------------------------------------------------------
# Cached resources (created once per Streamlit session, not per search call)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def _load_model() -> SentenceTransformer:
    """Load the sentence-transformer once and cache it for the whole app lifetime."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Connecting to Qdrant…")
def _vec_client() -> QdrantClient:
    """
    Create a single Qdrant client for the app using gRPC for lower latency.

    CHANGE vs src/search_vector.py:
      search_vector() calls _qdrant_client() on every invocation, which builds
      a new HTTP client each time using QDRANT_URL.  Here we build ONE gRPC client
      (prefer_grpc=True, grpc_port=6334) and keep it alive for the session.
      host and grpc_port are still read from .env / defaults — no hardcoded values.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    # Extract the hostname from the URL (strip scheme)
    host = qdrant_url.replace("http://", "").replace("https://", "").split(":")[0]
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(
        host=host,
        port=6333,
        grpc_port=6334,
        prefer_grpc=True,
        api_key=api_key,
    )


def _pg_connect():
    """Open a fresh PostgreSQL connection from .env."""
    url = os.getenv("POSTGRES_URL")
    if not url:
        raise RuntimeError("Missing POSTGRES_URL in .env")
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Vector search (calls the cached client + model; does the PG provenance join)
# ---------------------------------------------------------------------------

def _vector_search(query: str, subject: str | None) -> tuple[list[dict[str, Any]], float]:
    """
    Embed the query, search Qdrant, then join PostgreSQL to get file_name and page_no.

    Returns (results_list, elapsed_ms).
    Each result dict has: file_name, page_no, chunk_text, score.

    WHY the PG join:
      Qdrant payload stores {doc_id, subject, chunk_index, text} — no file_name or
      page_no.  We JOIN chunks + documents on doc_id + chunk_text to recover provenance.
    """
    model = _load_model()
    client = _vec_client()

    # Build subject filter for Qdrant if the user chose one
    query_filter = None
    if subject:
        query_filter = Filter(
            must=[FieldCondition(key="subject", match=MatchValue(value=subject))]
        )

    t0 = time.perf_counter()

    # Embed the query
    vector = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    # Search Qdrant
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        search_params=SearchParams(hnsw_ef=128),
        limit=TOP_K,
        with_payload=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Build raw results from Qdrant payload
    raw = [
        {
            "doc_id": (pt.payload or {}).get("doc_id"),
            "chunk_text": (pt.payload or {}).get("text", ""),
            "score": pt.score,
        }
        for pt in response.points
    ]

    # Join PostgreSQL to get file_name and page_no (not in Qdrant payload)
    enriched = []
    if raw:
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                for r in raw:
                    cur.execute(
                        """
                        SELECT d.file_name, c.page_no
                        FROM chunks c
                        JOIN documents d ON d.doc_id = c.doc_id
                        WHERE c.doc_id = %s AND c.chunk_text = %s
                        LIMIT 1
                        """,
                        (r["doc_id"], r["chunk_text"]),
                    )
                    row = cur.fetchone()
                    enriched.append(
                        {
                            "file_name": row[0] if row else "unknown",
                            "page_no":   row[1] if row else "?",
                            "chunk_text": r["chunk_text"],
                            "score": r["score"],
                        }
                    )
        finally:
            conn.close()

    return enriched, elapsed_ms


# ---------------------------------------------------------------------------
# Subject list from Postgres (for the filter dropdown)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _get_subjects() -> list[str]:
    """
    Read the distinct subjects stored in documents.
    Cached for 30 s so a freshly ingested PDF appears without a full reload.
    """
    try:
        conn = _pg_connect()
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT subject FROM documents ORDER BY subject")
            rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Streamlit page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Semantic Notes Search", layout="wide")
st.title("📚 Semantic Notes Search Engine")
st.caption("Compare SQL LIKE, SQL Full-Text Search and Qdrant Vector Search side-by-side.")

# ---------------------------------------------------------------------------
# SIDEBAR — PDF ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📥 Ingest a PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    subject_input = st.text_input("Subject (e.g. DBMS, Java, Comp)")
    user_id_input = st.number_input("User ID", min_value=1, value=1, step=1)

    if st.button("Ingest", disabled=uploaded is None):
        if not subject_input.strip():
            st.error("Please enter a subject before ingesting.")
        else:
            # Save to a temp folder using the REAL file name so PostgreSQL stores
            # the correct file_name (ingest_pdf uses Path(path).name).
            tmp_dir = Path(tempfile.mkdtemp())
            tmp_path = tmp_dir / uploaded.name          # preserves original name
            tmp_path.write_bytes(uploaded.read())

            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    result = ingest_pdf(str(tmp_path), subject_input.strip(), int(user_id_input))
                    st.success(
                        f"✅ Ingested **{uploaded.name}**\n\n"
                        f"- Chunks: {result['chunk_count']}\n"
                        f"- Job status: **{result['job_status']}**\n"
                        f"- Qdrant points: {result['qdrant_point_count']}"
                    )
                    # Invalidate subject cache so the new subject appears in the dropdown
                    _get_subjects.clear()
                except Exception as exc:
                    st.error(f"❌ Ingestion failed: {exc}")
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# MAIN AREA — search bar + filter + results
# ---------------------------------------------------------------------------
subjects = _get_subjects()

col_q, col_filter = st.columns([3, 1])
with col_q:
    query = st.text_input("🔍 Search query", placeholder="e.g. what is atomicity?")
with col_filter:
    subject_options = ["(all)"] + subjects
    chosen_subject = st.selectbox("Subject filter", subject_options)

search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        # Resolve subject: None means no filter
        active_subject = None if chosen_subject == "(all)" else chosen_subject

        # --- Run all three searches ---
        # 1. SQL LIKE — timed around the call
        t0 = time.perf_counter()
        try:
            like_results = search_like(query, limit=TOP_K)
        except Exception as exc:
            like_results = []
            like_error = str(exc)
        else:
            like_error = None
        like_ms = (time.perf_counter() - t0) * 1000

        # 2. SQL FTS — timed around the call
        t0 = time.perf_counter()
        try:
            fts_results = search_fts(query, limit=TOP_K)
        except Exception as exc:
            fts_results = []
            fts_error = str(exc)
        else:
            fts_error = None
        fts_ms = (time.perf_counter() - t0) * 1000

        # 3. Vector (gRPC client + PG join for file/page provenance)
        try:
            vec_results, vec_ms = _vector_search(query, active_subject)
            vec_error = None
        except Exception as exc:
            vec_results = []
            vec_ms = 0.0
            vec_error = str(exc)

        # --- Display ---
        col1, col2, col3 = st.columns(3)

        # Helper to render one result row
        def _render_result(r: dict[str, Any], show_score: bool = False) -> None:
            file_name = r.get("file_name", "unknown")
            page_no   = r.get("page_no", "?")
            text      = r.get("chunk_text", r.get("text", ""))
            snippet   = text[:SNIPPET_LEN] + ("…" if len(text) > SNIPPET_LEN else "")
            label = f"📄 **{file_name}** — Page {page_no}"
            if show_score:
                label += f"  `score={r['score']:.4f}`"
            st.markdown(label)
            st.caption(snippet)
            st.markdown("---")

        # 1. SQL LIKE column (red)
        with col1:
            st.markdown("## :red[SQL LIKE]")
            st.markdown(f"⏱ `{like_ms:.1f} ms` &nbsp;|&nbsp; `{len(like_results)} result(s)`")
            if like_error:
                st.error(like_error)
            elif not like_results:
                st.info("No results — LIKE found no substring match.")
            else:
                for r in like_results:
                    _render_result(r)

        # 2. SQL FTS column (yellow / orange)
        with col2:
            st.markdown("## :orange[SQL Full-Text]")
            st.markdown(f"⏱ `{fts_ms:.1f} ms` &nbsp;|&nbsp; `{len(fts_results)} result(s)`")
            if fts_error:
                st.error(fts_error)
            elif not fts_results:
                st.info("No results — FTS found no lexeme match.")
            else:
                for r in fts_results:
                    _render_result(r)

        # 3. Vector column (green)
        with col3:
            filter_label = f" (subject={active_subject})" if active_subject else ""
            st.markdown("## :green[Vector Search]" + filter_label)
            st.markdown(f"⏱ `{vec_ms:.1f} ms` &nbsp;|&nbsp; `{len(vec_results)} result(s)`")
            if vec_error:
                st.error(vec_error)
            elif not vec_results:
                st.info("No results — no semantically similar chunks found.")
            else:
                for r in vec_results:
                    _render_result(r, show_score=True)
