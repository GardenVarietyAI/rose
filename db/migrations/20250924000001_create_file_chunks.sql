-- migrate:up
-- Store file chunks with embeddings
CREATE TABLE IF NOT EXISTS file_chunks (
    content_hash TEXT PRIMARY KEY,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    meta JSON,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_chunks_file_id ON file_chunks(file_id);

-- migrate:down
DROP TABLE IF EXISTS file_chunks;
ALTER TABLE files DROP COLUMN processing_status;
