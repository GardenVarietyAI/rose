-- migrate:up
CREATE TABLE fine_tuning_jobs (
    id TEXT PRIMARY KEY,
    organization_id TEXT,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    finished_at INTEGER,
    training_file TEXT NOT NULL,
    validation_file TEXT,
    result_files JSON,
    trained_tokens INTEGER,
    error JSON,
    fine_tuned_model TEXT,
    seed INTEGER,
    suffix TEXT,
    meta JSON,
    hyperparameters JSON,
    method JSON,
    trainer TEXT,
    training_metrics JSON
);

CREATE INDEX idx_fine_tuning_jobs_status ON fine_tuning_jobs(status);
CREATE INDEX idx_fine_tuning_jobs_created_at ON fine_tuning_jobs(created_at);

-- migrate:down
DROP INDEX IF EXISTS idx_fine_tuning_jobs_created_at;
DROP INDEX IF EXISTS idx_fine_tuning_jobs_status;
DROP TABLE fine_tuning_jobs;
