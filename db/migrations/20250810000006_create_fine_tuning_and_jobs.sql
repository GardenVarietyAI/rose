-- migrate:up
-- Create fine_tuning_jobs table
CREATE TABLE fine_tuning_jobs (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    finished_at INTEGER,
    model TEXT NOT NULL,
    fine_tuned_model TEXT,
    organization_id TEXT NOT NULL DEFAULT 'org-local',
    result_files TEXT NOT NULL DEFAULT '[]',  -- JSON
    status TEXT NOT NULL,
    validation_file TEXT,
    training_file TEXT NOT NULL,
    error TEXT,  -- JSON, nullable
    seed INTEGER NOT NULL DEFAULT 42,
    trained_tokens INTEGER,
    meta TEXT,  -- JSON, nullable
    started_at INTEGER,
    suffix TEXT,
    hyperparameters TEXT NOT NULL DEFAULT '{}',  -- JSON
    method TEXT,  -- JSON, nullable
    trainer TEXT NOT NULL DEFAULT 'huggingface',
    training_metrics TEXT  -- JSON, nullable
);

-- Create indexes for fine_tuning_jobs
CREATE INDEX idx_ft_jobs_status ON fine_tuning_jobs(status);
CREATE INDEX idx_ft_jobs_created ON fine_tuning_jobs(created_at);

-- Create fine_tuning_events table
CREATE TABLE fine_tuning_events (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL DEFAULT 'fine_tuning.job.event',
    created_at INTEGER NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,  -- JSON, nullable
    job_id TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES fine_tuning_jobs(id) ON DELETE CASCADE
);

-- Create indexes for fine_tuning_events
CREATE INDEX idx_events_job_id ON fine_tuning_events(job_id);
CREATE INDEX idx_events_created ON fine_tuning_events(created_at);

-- Create jobs table
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,  -- JSON
    result TEXT,  -- JSON, nullable
    error TEXT,
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3
);

-- Create indexes for jobs
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_type_status ON jobs(type, status);

-- migrate:down
DROP INDEX IF EXISTS idx_jobs_type_status;
DROP INDEX IF EXISTS idx_jobs_status;
DROP TABLE IF EXISTS jobs;

DROP INDEX IF EXISTS idx_events_created;
DROP INDEX IF EXISTS idx_events_job_id;
DROP TABLE IF EXISTS fine_tuning_events;

DROP INDEX IF EXISTS idx_ft_jobs_created;
DROP INDEX IF EXISTS idx_ft_jobs_status;
DROP TABLE fine_tuning_jobs;
