-- migrate:up
CREATE TABLE assistants (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'assistant',
    created_at INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    model TEXT NOT NULL,
    instructions TEXT,
    tools TEXT NOT NULL DEFAULT '[]',  -- JSON
    tool_resources TEXT NOT NULL DEFAULT '{}',  -- JSON
    meta TEXT NOT NULL DEFAULT '{}',  -- JSON
    temperature REAL DEFAULT 0.7,
    top_p REAL DEFAULT 0.8,
    response_format TEXT DEFAULT NULL  -- JSON
);

-- Create indexes
CREATE INDEX idx_assistants_created ON assistants(created_at);
CREATE INDEX idx_assistants_model ON assistants(model);

-- migrate:down
DROP INDEX IF EXISTS idx_assistants_model;
DROP INDEX IF EXISTS idx_assistants_created;
DROP TABLE assistants;
