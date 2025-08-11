-- migrate:up
-- Create message_metadata table (moved from migration 003 to fix foreign key dependency)
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
DROP TABLE message_metadata;
