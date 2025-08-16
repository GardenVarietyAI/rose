-- migrate:up

-- Create vector_store_files table for tracking file processing status
CREATE TABLE vector_store_files (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'vector_store.file',
    vector_store_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',
    created_at INTEGER NOT NULL,
    last_error JSON,
    FOREIGN KEY (vector_store_id) REFERENCES vector_stores (id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
);

-- Add indexes for efficient queries
CREATE INDEX idx_vector_store_files_store_id ON vector_store_files(vector_store_id);
CREATE INDEX idx_vector_store_files_file_id ON vector_store_files(file_id);
CREATE UNIQUE INDEX idx_vector_store_files_unique ON vector_store_files(vector_store_id, file_id);

-- migrate:down
DROP INDEX IF EXISTS idx_vector_store_files_store_id;
DROP INDEX IF EXISTS idx_vector_store_files_file_id;
DROP INDEX IF EXISTS idx_vector_store_files_unique;
DROP TABLE IF EXISTS vector_store_files;
