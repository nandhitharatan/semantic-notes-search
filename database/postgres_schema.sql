CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE documents (
    doc_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (user_id),
    file_name TEXT NOT NULL,
    subject TEXT
);

CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE TABLE ingestion_jobs (
    job_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents (doc_id),
    status VARCHAR(50) NOT NULL,
    error_message TEXT
);

CREATE INDEX idx_chunks_chunk_text_fts
    ON chunks
    USING GIN (to_tsvector('english', chunk_text));

INSERT INTO users (user_id, username, email)
VALUES (1, 'dummy_user', 'dummy@example.com');

SELECT setval(
    pg_get_serial_sequence('users', 'user_id'),
    (SELECT MAX(user_id) FROM users)
);
