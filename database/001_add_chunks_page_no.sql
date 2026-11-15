-- Preserve the source page for result validation and the Step 7 question runner.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS page_no INTEGER;
