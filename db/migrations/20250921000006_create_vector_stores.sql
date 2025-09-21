-- migrate:up
CREATE TABLE vector_stores (
    id TEXT PRIMARY KEY,
    object TEXT,
    name TEXT NOT NULL,
    dimensions INTEGER,
    meta JSON,
    created_at INTEGER NOT NULL,
    last_used_at INTEGER
);

CREATE INDEX idx_vector_stores_created_at ON vector_stores(created_at);
CREATE INDEX idx_vector_stores_name ON vector_stores(name);

-- migrate:down
DROP INDEX IF EXISTS idx_vector_stores_name;
DROP INDEX IF EXISTS idx_vector_stores_created_at;
DROP TABLE vector_stores;
