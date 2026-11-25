-- RAG document store (pgvector). Safe for fresh databases and for upgrades
-- from the former app/init_db.py + app/migrate_v03.py bootstrap scripts.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT,
    content_type TEXT,
    size_bytes BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    UNIQUE (document_id, chunk_index)
);

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS sha256 TEXT,
    ADD COLUMN IF NOT EXISTS content_type TEXT,
    ADD COLUMN IF NOT EXISTS size_bytes BIGINT;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS page_number INTEGER;

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha256
    ON documents (sha256)
    WHERE sha256 IS NOT NULL;
