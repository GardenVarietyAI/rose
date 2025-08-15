-- migrate:up
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'thread.message',
    created_at INTEGER NOT NULL,
    thread_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,  -- JSON
    assistant_id TEXT,
    run_id TEXT,
    attachments TEXT NOT NULL DEFAULT '[]',  -- JSON
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON
    response_chain_id TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    incomplete_details TEXT,  -- JSON, nullable
    incomplete_at INTEGER,
    completed_at INTEGER,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX idx_messages_thread ON messages(thread_id);
CREATE INDEX idx_messages_created ON messages(created_at);
CREATE INDEX idx_messages_role ON messages(role);
CREATE INDEX idx_messages_response_chain ON messages(response_chain_id);

-- migrate:down
DROP INDEX IF EXISTS idx_messages_response_chain;
DROP INDEX IF EXISTS idx_messages_role;
DROP INDEX IF EXISTS idx_messages_created;
DROP INDEX IF EXISTS idx_messages_thread;
DROP TABLE messages;
