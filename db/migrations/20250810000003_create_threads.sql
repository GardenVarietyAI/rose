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

-- Create message_metadata table
CREATE TABLE message_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- Create indexes for message_metadata
CREATE INDEX idx_message_metadata_message ON message_metadata(message_id);
CREATE INDEX idx_message_metadata_key ON message_metadata(key);

-- migrate:down
DROP INDEX IF EXISTS idx_message_metadata_key;
DROP INDEX IF EXISTS idx_message_metadata_message;
DROP TABLE IF EXISTS message_metadata;
DROP INDEX IF EXISTS idx_threads_created;
DROP TABLE threads;
