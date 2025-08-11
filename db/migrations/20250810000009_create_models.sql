-- migrate:up
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    name TEXT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL DEFAULT 'huggingface',
    path TEXT,
    is_fine_tuned INTEGER NOT NULL DEFAULT 0,  -- Boolean as integer

    -- Model parameters
    temperature REAL NOT NULL DEFAULT 0.7,
    top_p REAL NOT NULL DEFAULT 0.9,
    memory_gb REAL NOT NULL DEFAULT 2.0,
    timeout INTEGER,
    quantization TEXT,

    -- LoRA configuration
    lora_target_modules TEXT,  -- JSON array

    -- OpenAI API compatibility fields
    owned_by TEXT NOT NULL DEFAULT 'organization-owner',
    parent TEXT,
    permissions TEXT DEFAULT '[]',  -- JSON array

    -- Metadata
    created_at INTEGER NOT NULL
);

-- Create indexes
CREATE INDEX idx_models_model_name ON models(model_name);
CREATE INDEX idx_models_is_fine_tuned ON models(is_fine_tuned);

-- migrate:down
DROP INDEX IF EXISTS idx_models_is_fine_tuned;
DROP INDEX IF EXISTS idx_models_model_name;
DROP TABLE models;
