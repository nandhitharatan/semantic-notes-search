-- Count delivery attempts so a permanently failing outbox job can be marked FAILED.
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0;
