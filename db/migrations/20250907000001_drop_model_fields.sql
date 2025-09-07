-- migrate:up
ALTER TABLE models DROP COLUMN name;
ALTER TABLE models DROP COLUMN model_type;
ALTER TABLE models DROP COLUMN memory_gb;

-- migrate:down
ALTER TABLE models ADD COLUMN name TEXT;
ALTER TABLE models ADD COLUMN name TEXT DEFAULT 'huggingface';
ALTER TABLE models ADD COLUMN memory_gb REAL NOT NULL DEFAULT 2.0;
