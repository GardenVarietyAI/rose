-- migrate:up
CREATE TABLE IF NOT EXISTS vector_store_files (
    id TEXT PRIMARY KEY,
    object TEXT,
    vector_store_id TEXT NOT NULL REFERENCES vector_stores(id),
    file_id TEXT NOT NULL,
    status TEXT,
    last_error JSON,
    created_at INTEGER NOT NULL,
    UNIQUE (vector_store_id, file_id)
);

CREATE INDEX IF NOT EXISTS idx_vector_store_files_vector_store_id ON vector_store_files(vector_store_id);
CREATE INDEX IF NOT EXISTS idx_vector_store_files_file_id ON vector_store_files(file_id);

-- migrate:down
DROP INDEX IF EXISTS idx_vector_store_files_file_id;
DROP INDEX IF EXISTS idx_vector_store_files_vector_store_id;
DROP TABLE vector_store_files;
