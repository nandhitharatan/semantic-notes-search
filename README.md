# Semantic Notes Search Engine

Students upload PDF notes and search their contents by meaning. PostgreSQL stores
the durable document and chunk records, while Qdrant stores embeddings for semantic
search.

## Local setup

1. Start the database services:

   ```powershell
   docker compose up -d
   ```

2. Install the fixed project dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. On a new database, apply the base schema and then the page-number migration:

   ```powershell
   Get-Content database/postgres_schema.sql -Raw | docker compose exec -T postgres psql -U admin -d notes_db
   Get-Content database/001_add_chunks_page_no.sql -Raw | docker compose exec -T postgres psql -U admin -d notes_db
   ```

4. Create the Qdrant collection:

   ```powershell
   py database/qdrant_setup.py
   ```

`POSTGRES_URL`, `QDRANT_URL`, and `QDRANT_API_KEY` are read from the local `.env`
file. Do not commit secrets.

## Step 1: ingest a PDF

Run `ingest.py` with a PDF path, subject, and existing PostgreSQL user ID:

```powershell
py src/ingest.py "C:\path\to\your-notes.pdf" "DBMS" 1
```

The function commits the document, chunks, page numbers, and a `PENDING` ingestion
job to PostgreSQL before sending vectors to Qdrant. A Qdrant failure therefore leaves
the PostgreSQL data and `PENDING` job available for the retry worker added in Step 4.
On success, the script prints the number of chunks, PostgreSQL job status, and number
of Qdrant points for the new document.

## Ingest a subject folder

Place PDFs in `data\Java`, `data\DBMS`, or `data\Comp` (file names may contain
spaces), then ingest all of them with the default PostgreSQL user ID `1`:

```powershell
py src/ingest_folder.py data
```

To use another user, pass `--user-id`, for example:

```powershell
py src/ingest_folder.py data --user-id 2
```

Files that already have the same file name and subject in PostgreSQL are printed as
`skipped`; this prevents duplicate relational rows and Qdrant points on later runs.

## Step 2: PostgreSQL searches

Run both SQL search methods from the project root:

```powershell
py -c "from src.search_sql import search_like, search_fts; print(search_like('database', 5)); print(search_fts('database', 5))"
```

`search_like` is the substring baseline. `search_fts` uses PostgreSQL's English
full-text query and `ts_rank`, with the existing GIN index supporting the tsvector
predicate.

## Step 3: Qdrant vector search

Run semantic search with the default HNSW settings:

```powershell
py -c "from src.search_vector import search_vector; print(search_vector('database temperature', 5))"
```

Pass `subject='DBMS'` to filter by payload, `exact=True` for exact search, or an
`hnsw_ef` value to control query-time HNSW exploration.

## Step 4: retry worker and recovery proof

Apply the retry-count migration once:

```powershell
Get-Content database/002_add_ingestion_retry_count.sql -Raw | docker compose exec -T postgres psql -U admin -d notes_db
```

Run the worker to re-embed every retryable `PENDING` job and idempotently upsert its
stable chunk UUIDs into Qdrant. It prints embedding and Qdrant-upsert times separately:

```powershell
py database/retry_worker.py
```

To prove failure and recovery, provide a small PDF, manually stop Qdrant when prompted,
then start it again when prompted. The script verifies `PENDING` before recovery and
`COMPLETED` after the worker runs:

```powershell
py database/test_failure_recovery.py "data\DBMS\INDEX, JOINS.pdf"
```
