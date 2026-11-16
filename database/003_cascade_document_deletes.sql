BEGIN;

ALTER TABLE chunks
    DROP CONSTRAINT chunks_doc_id_fkey,
    ADD CONSTRAINT chunks_doc_id_fkey
        FOREIGN KEY (doc_id)
        REFERENCES documents (doc_id)
        ON DELETE CASCADE;

ALTER TABLE ingestion_jobs
    DROP CONSTRAINT ingestion_jobs_doc_id_fkey,
    ADD CONSTRAINT ingestion_jobs_doc_id_fkey
        FOREIGN KEY (doc_id)
        REFERENCES documents (doc_id)
        ON DELETE CASCADE;

COMMIT;
