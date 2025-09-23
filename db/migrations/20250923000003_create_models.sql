-- migrate:up
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    path TEXT,
    is_fine_tuned INTEGER,
    kind TEXT,
    temperature REAL,
    top_p REAL,
    timeout INTEGER,
    quantization TEXT,
    lora_target_modules JSON,
    owned_by TEXT,
    parent TEXT,
    permissions JSON,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_model_name ON models(model_name);
CREATE INDEX IF NOT EXISTS idx_models_is_fine_tuned ON models(is_fine_tuned);

-- migrate:down
DROP INDEX IF EXISTS idx_models_is_fine_tuned;
DROP INDEX IF EXISTS idx_models_model_name;
DROP TABLE models;
