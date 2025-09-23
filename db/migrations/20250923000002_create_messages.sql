-- migrate:up
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    object TEXT,
    role TEXT NOT NULL,
    content JSON NOT NULL,
    attachments JSON,
    meta JSON,
    status TEXT,
    created_at INTEGER NOT NULL,
    response_chain_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_response_chain ON messages(response_chain_id);

-- migrate:down
DROP INDEX IF EXISTS idx_messages_response_chain;
DROP INDEX IF EXISTS idx_messages_created_at;
DROP TABLE messages;
