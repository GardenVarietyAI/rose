-- migrate:up
-- Store file chunks with embeddings
CREATE TABLE IF NOT EXISTS file_chunks (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    meta JSON,
    created_at INTEGER NOT NULL,
    UNIQUE(file_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_file_chunks_file_id ON file_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_file_chunks_content_hash ON file_chunks(content_hash);

-- migrate:down
DROP TABLE IF EXISTS file_chunks;
