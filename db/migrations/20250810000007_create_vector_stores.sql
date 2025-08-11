-- migrate:up
-- Create vector_stores table for metadata
CREATE TABLE vector_stores (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'vector_store',
    name TEXT NOT NULL,
    dimensions INTEGER NOT NULL DEFAULT 384,
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON for additional metadata
    created_at INTEGER NOT NULL
);

-- Create indexes for vector_stores
CREATE INDEX idx_vector_stores_created_at ON vector_stores(created_at);
CREATE INDEX idx_vector_stores_name ON vector_stores(name);

-- Add vector_store_id column to files table
ALTER TABLE files ADD COLUMN vector_store_id TEXT REFERENCES vector_stores(id);

-- Create index for the new foreign key
CREATE INDEX idx_files_vector_store_id ON files(vector_store_id);

-- migrate:down
DROP INDEX IF EXISTS idx_files_vector_store_id;
ALTER TABLE files DROP COLUMN vector_store_id;
DROP INDEX IF EXISTS idx_vector_stores_name;
DROP INDEX IF EXISTS idx_vector_stores_created_at;
DROP TABLE vector_stores;
