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

Call `ingest_pdf` with a PDF path, subject, and existing PostgreSQL user ID:

```powershell
py -c "from src.ingest import ingest_pdf; print(ingest_pdf('path/to/notes.pdf', 'DBMS', 1))"
```

The function commits the document, chunks, page numbers, and a `PENDING` ingestion
job to PostgreSQL before sending vectors to Qdrant. A Qdrant failure therefore leaves
the PostgreSQL data and `PENDING` job available for the retry worker added in Step 4.
