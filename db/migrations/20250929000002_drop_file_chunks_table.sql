-- migrate:up
DROP TABLE IF EXISTS file_chunks;

-- migrate:down
-- Recreate file_chunks table
CREATE TABLE file_chunks (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    meta JSON,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_file_chunks_file_id ON file_chunks(file_id);
CREATE INDEX idx_file_chunks_content_hash ON file_chunks(content_hash);

-- Re-add status column to files table
ALTER TABLE files ADD COLUMN status TEXT DEFAULT 'uploaded';
