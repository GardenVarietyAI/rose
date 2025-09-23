-- migrate:up
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    vector_store_id TEXT NOT NULL REFERENCES vector_stores(id),
    file_id TEXT,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    meta JSON,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_vector_store_id ON documents(vector_store_id);
CREATE INDEX IF NOT EXISTS idx_documents_file_id ON documents(file_id);
CREATE INDEX IF NOT EXISTS idx_documents_vector_store_file ON documents(vector_store_id, file_id);

-- migrate:down
DROP INDEX IF EXISTS idx_documents_vector_store_file;
DROP INDEX IF EXISTS idx_documents_file_id;
DROP INDEX IF EXISTS idx_documents_vector_store_id;
DROP TABLE documents;
