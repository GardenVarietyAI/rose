-- migrate:up

-- Create documents table for text chunks
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    vector_store_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    meta JSON,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (vector_store_id) REFERENCES vector_stores (id) ON DELETE CASCADE
);

-- Add indexes for vector_store_id queries
CREATE INDEX idx_documents_store_id ON documents(vector_store_id);
CREATE INDEX idx_documents_store_created ON documents(vector_store_id, created_at);

-- Add last_used_at column to vector_stores
ALTER TABLE vector_stores ADD COLUMN last_used_at INTEGER;

-- Remove vector_store_id from files table (was not being used)
DROP INDEX IF EXISTS idx_files_vector_store_id;
ALTER TABLE files DROP COLUMN vector_store_id;

-- migrate:down
ALTER TABLE files ADD COLUMN vector_store_id TEXT REFERENCES vector_stores(id);
CREATE INDEX idx_files_vector_store_id ON files(vector_store_id);
ALTER TABLE vector_stores DROP COLUMN last_used_at;
DROP INDEX IF EXISTS idx_documents_store_id;
DROP TABLE IF EXISTS documents;
