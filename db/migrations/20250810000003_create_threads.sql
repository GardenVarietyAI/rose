-- migrate:up
CREATE TABLE threads (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'thread',
    created_at INTEGER NOT NULL,
    tool_resources TEXT,  -- JSON, nullable
    meta TEXT NOT NULL DEFAULT '{}'  -- JSON
);

-- Create indexes
CREATE INDEX idx_threads_created ON threads(created_at);

-- migrate:down
DROP INDEX IF EXISTS idx_threads_created;
DROP TABLE threads;
