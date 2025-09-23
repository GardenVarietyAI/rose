-- migrate:up
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    object TEXT,
    bytes INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    filename TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT,
    expires_at INTEGER,
    status_details TEXT,
    content BLOB,
    storage_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at);
CREATE INDEX IF NOT EXISTS idx_files_purpose ON files(purpose);

-- migrate:down
DROP INDEX IF EXISTS idx_files_purpose;
DROP INDEX IF EXISTS idx_files_created_at;
DROP TABLE files;
