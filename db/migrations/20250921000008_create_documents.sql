-- migrate:up
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    vector_store_id TEXT NOT NULL REFERENCES vector_stores(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    meta JSON
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_documents_vector_store_id ON documents(vector_store_id);
CREATE INDEX idx_documents_created_at ON documents(created_at);

-- migrate:down
DROP INDEX IF EXISTS idx_documents_created_at;
DROP INDEX IF EXISTS idx_documents_vector_store_id;
DROP TABLE documents;
