-- migrate:up
CREATE TABLE files (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'file',
    bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    filename TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT DEFAULT 'processed',
    status_details TEXT,
    storage_path TEXT NOT NULL
);

-- Create indexes for common queries
CREATE INDEX idx_files_created_at ON files(created_at);
CREATE INDEX idx_files_purpose ON files(purpose);
CREATE INDEX idx_files_status ON files(status);

-- migrate:down
DROP INDEX IF EXISTS idx_files_status;
DROP INDEX IF EXISTS idx_files_purpose;
DROP INDEX IF EXISTS idx_files_created_at;
DROP TABLE files;
